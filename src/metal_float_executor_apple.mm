#include "metal_float_executor_apple.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <os/signpost.h>

#include <dispatch/dispatch.h>

#include "dsmvc_metal_routes_metallib.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace dsmvc::experimental {
namespace {

struct AxisJob {
    std::uint32_t source_size = 0U;
    std::uint32_t destination_size = 0U;
    std::uint32_t vector_count = 0U;
    std::uint32_t input_stride = 0U;
    std::uint32_t output_stride = 0U;
    std::uint32_t direction = 0U;
    std::uint32_t half_bandwidth = 0U;
    std::uint32_t reserved = 0U;
    std::uint32_t batch_count = 0U;
    std::uint32_t input_frame_stride = 0U;
    std::uint32_t output_frame_stride = 0U;
    std::uint32_t reserved_2 = 0U;
};

static_assert(sizeof(AxisJob) == 48U);

struct PlanBuffers {
    id<MTLBuffer> transpose_offsets = nil;
    id<MTLBuffer> transpose_indices = nil;
    id<MTLBuffer> transpose_weights = nil;
    id<MTLBuffer> lower_ld = nil;
    id<MTLBuffer> upper_l = nil;
    id<MTLBuffer> inverse_diagonal = nil;
    std::uint32_t half_bandwidth = 0U;
};

[[nodiscard]] constexpr std::uint32_t align_up(
    std::uint32_t value, std::uint32_t alignment) noexcept {
    return (value + alignment - 1U) / alignment * alignment;
}

[[nodiscard]] std::string ns_error(NSError *error, const char *fallback) {
    if (error == nil || error.localizedDescription == nil) return fallback;
    const char *description = error.localizedDescription.UTF8String;
    return description == nullptr ? fallback : description;
}

[[nodiscard]] os_log_t profile_log() noexcept {
    static os_log_t log = os_log_create(
        "com.dsmvc.plugin", OS_LOG_CATEGORY_POINTS_OF_INTEREST);
    return log;
}

void copy_strided_rows(
    void *destination, std::ptrdiff_t destination_stride,
    const void *source, std::ptrdiff_t source_stride, std::size_t row_bytes,
    std::uint32_t row_count, MetalFloatStagingStats &stats) {
    if (row_count == 0U) return;
    if (destination_stride < static_cast<std::ptrdiff_t>(row_bytes)
        || source_stride < static_cast<std::ptrdiff_t>(row_bytes)) {
        throw std::invalid_argument("invalid GRAYS Metal plane stride");
    }

    auto *destination_bytes = static_cast<std::byte *>(destination);
    const auto *source_bytes = static_cast<const std::byte *>(source);
    if (destination_stride == source_stride) {
        const std::size_t copied_bytes =
            (static_cast<std::size_t>(row_count) - 1U)
                * static_cast<std::size_t>(source_stride)
            + row_bytes;
        std::memcpy(destination_bytes, source_bytes, copied_bytes);
        ++stats.memcpy_calls;
        stats.copied_bytes += copied_bytes;
        return;
    }

    for (std::uint32_t row = 0U; row < row_count; ++row) {
        std::memcpy(
            destination_bytes + static_cast<std::ptrdiff_t>(row)
                * destination_stride,
            source_bytes + static_cast<std::ptrdiff_t>(row) * source_stride,
            row_bytes);
    }
    stats.memcpy_calls += row_count;
    stats.copied_bytes += static_cast<std::size_t>(row_count) * row_bytes;
}

} // namespace

struct MetalFloatExecutor::Impl {
    Impl(std::shared_ptr<const AxisPlan> requested_horizontal,
         std::shared_ptr<const AxisPlan> requested_vertical,
         std::size_t requested_maximum_batch_size,
         std::size_t requested_threads_per_threadgroup,
         bool requested_profile_signposts)
        : horizontal(std::move(requested_horizontal)),
          vertical(std::move(requested_vertical)),
          maximum_batch_size(requested_maximum_batch_size),
          threads_per_threadgroup(requested_threads_per_threadgroup),
          profile_signposts(requested_profile_signposts) {
        if (!horizontal || !vertical || !horizontal->valid() || !vertical->valid()) {
            throw std::invalid_argument("experimental GRAYS Metal plan is invalid");
        }
        if (horizontal->requires_float64() || vertical->requires_float64()) {
            throw std::invalid_argument(
                "experimental GRAYS Metal executor supports F32 plans only");
        }
        if (maximum_batch_size == 0U || maximum_batch_size > 64U
            || threads_per_threadgroup == 0U
            || threads_per_threadgroup > 1024U) {
            throw std::invalid_argument("invalid experimental GRAYS Metal configuration");
        }

        source_width = static_cast<std::uint32_t>(horizontal->source_size);
        destination_width = static_cast<std::uint32_t>(horizontal->destination_size);
        source_height = static_cast<std::uint32_t>(vertical->source_size);
        destination_height = static_cast<std::uint32_t>(vertical->destination_size);
        input_stride = align_up(source_width * sizeof(float), 64U) / sizeof(float);
        intermediate_stride =
            align_up(destination_width * sizeof(float), 64U) / sizeof(float);
        output_stride =
            align_up(destination_width * sizeof(float), 32U) / sizeof(float);

        device = MTLCreateSystemDefaultDevice();
        if (device == nil || !device.hasUnifiedMemory) {
            throw std::runtime_error(
                "experimental GRAYS Metal executor requires unified memory");
        }
        const char *device_text = device.name.UTF8String;
        name = device_text == nullptr ? "Metal device" : device_text;
        queue = [device newCommandQueue];
        if (queue == nil) throw std::runtime_error("Metal command queue creation failed");

        dispatch_data_t data = dispatch_data_create(
            dsmvc_metal_routes_metallib, dsmvc_metal_routes_metallib_size,
            dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
            DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        NSError *error = nil;
        library = [device newLibraryWithData:data error:&error];
        if (library == nil) {
            throw std::runtime_error(ns_error(error, "embedded metallib load failed"));
        }

        const std::array<NSString *, 5> names{
            @"inverse_axis_generic", @"inverse_axis_h1", @"inverse_axis_h3",
            @"inverse_axis_h5", @"inverse_axis_h7"};
        for (std::size_t index = 0U; index < names.size(); ++index) {
            pipelines[index] = make_pipeline(names[index]);
        }
        horizontal_buffers = prepare_plan(*horizontal, @"horizontal");
        vertical_buffers = prepare_plan(*vertical, @"vertical");

        const std::size_t input_bytes = static_cast<std::size_t>(source_height)
            * input_stride * sizeof(float) * maximum_batch_size;
        const std::size_t intermediate_bytes = static_cast<std::size_t>(source_height)
            * intermediate_stride * sizeof(float) * maximum_batch_size;
        const std::size_t output_bytes = static_cast<std::size_t>(destination_height)
            * output_stride * sizeof(float) * maximum_batch_size;
        input = make_empty_buffer(input_bytes, @"dsmvc plugin GRAYS input");
        intermediate = make_empty_buffer(
            intermediate_bytes, @"dsmvc plugin GRAYS intermediate");
        output = make_empty_buffer(output_bytes, @"dsmvc plugin GRAYS output");
    }

    [[nodiscard]] id<MTLComputePipelineState> make_pipeline(
        NSString *function_name) {
        id<MTLFunction> function = [library newFunctionWithName:function_name];
        if (function == nil) {
            throw std::runtime_error(
                "embedded metallib is missing "
                + std::string{function_name.UTF8String});
        }
        NSError *error = nil;
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&error];
        if (pipeline == nil) {
            throw std::runtime_error(ns_error(error, "Metal pipeline creation failed"));
        }
        return pipeline;
    }

    [[nodiscard]] id<MTLBuffer> make_empty_buffer(
        std::size_t bytes, NSString *label) {
        if (bytes == 0U || bytes > device.maxBufferLength) {
            throw std::length_error("invalid experimental GRAYS Metal buffer size");
        }
        id<MTLBuffer> buffer = [device newBufferWithLength:bytes
                                                  options:MTLResourceStorageModeShared];
        if (buffer == nil) throw std::runtime_error("Metal buffer allocation failed");
        buffer.label = label;
        requested_bytes += bytes;
        return buffer;
    }

    template <class Value>
    [[nodiscard]] id<MTLBuffer> make_plan_buffer(
        const std::vector<Value> &values, NSString *label) {
        if (values.empty()) {
            return make_empty_buffer(sizeof(std::uint32_t), label);
        }
        const std::size_t bytes = values.size() * sizeof(Value);
        id<MTLBuffer> buffer = [device newBufferWithBytes:values.data()
                                                  length:bytes
                                                 options:MTLResourceStorageModeShared];
        if (buffer == nil) throw std::runtime_error("Metal plan upload failed");
        buffer.label = label;
        requested_bytes += bytes;
        return buffer;
    }

    [[nodiscard]] PlanBuffers prepare_plan(
        const AxisPlan &plan, NSString *axis) {
        PlanBuffers prepared;
        prepared.transpose_offsets = make_plan_buffer(
            plan.transpose_offsets,
            [NSString stringWithFormat:@"dsmvc plugin GRAYS %@ offsets", axis]);
        prepared.transpose_indices = make_plan_buffer(
            plan.transpose_indices,
            [NSString stringWithFormat:@"dsmvc plugin GRAYS %@ indices", axis]);
        prepared.transpose_weights = make_plan_buffer(
            plan.transpose_weights,
            [NSString stringWithFormat:@"dsmvc plugin GRAYS %@ weights", axis]);
        prepared.lower_ld = make_plan_buffer(
            plan.lower_ld,
            [NSString stringWithFormat:@"dsmvc plugin GRAYS %@ lower LD", axis]);
        prepared.upper_l = make_plan_buffer(
            plan.upper_l,
            [NSString stringWithFormat:@"dsmvc plugin GRAYS %@ upper L", axis]);
        prepared.inverse_diagonal = make_plan_buffer(
            plan.inverse_diagonal,
            [NSString stringWithFormat:@"dsmvc plugin GRAYS %@ inverse diagonal", axis]);
        prepared.half_bandwidth = static_cast<std::uint32_t>(plan.half_bandwidth);
        return prepared;
    }

    [[nodiscard]] id<MTLComputePipelineState> pipeline(
        std::uint32_t half_bandwidth) const noexcept {
        switch (half_bandwidth) {
        case 1U: return pipelines[1];
        case 3U: return pipelines[2];
        case 5U: return pipelines[3];
        case 7U: return pipelines[4];
        default: return pipelines[0];
        }
    }

    void bind_plan(id<MTLComputeCommandEncoder> encoder,
                   const PlanBuffers &plan) {
        [encoder setBuffer:plan.transpose_offsets offset:0 atIndex:2];
        [encoder setBuffer:plan.transpose_indices offset:0 atIndex:3];
        [encoder setBuffer:plan.transpose_weights offset:0 atIndex:4];
        [encoder setBuffer:plan.lower_ld offset:0 atIndex:5];
        [encoder setBuffer:plan.upper_l offset:0 atIndex:6];
        [encoder setBuffer:plan.inverse_diagonal offset:0 atIndex:7];
    }

    void encode_axis(id<MTLCommandBuffer> command, id<MTLBuffer> source,
                     id<MTLBuffer> destination, const PlanBuffers &plan,
                     const AxisJob &job, NSString *label) {
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (encoder == nil) throw std::runtime_error("Metal encoder creation failed");
        encoder.label = label;
        id<MTLComputePipelineState> state = pipeline(plan.half_bandwidth);
        [encoder setComputePipelineState:state];
        [encoder setBuffer:source offset:0 atIndex:0];
        [encoder setBytes:&job length:sizeof(job) atIndex:1];
        bind_plan(encoder, plan);
        [encoder setBuffer:destination offset:0 atIndex:8];
        const NSUInteger threads = std::min<NSUInteger>(
            static_cast<NSUInteger>(threads_per_threadgroup),
            state.maxTotalThreadsPerThreadgroup);
        const std::size_t dispatch_count = static_cast<std::size_t>(job.vector_count)
            * job.batch_count;
        if (dispatch_count > std::numeric_limits<std::uint32_t>::max()) {
            throw std::length_error("GRAYS Metal dispatch exceeds 32-bit range");
        }
        [encoder dispatchThreads:MTLSizeMake(dispatch_count, 1U, 1U)
             threadsPerThreadgroup:MTLSizeMake(threads, 1U, 1U)];
        [encoder endEncoding];
    }

    void encode(id<MTLCommandBuffer> command, std::size_t frame_count) {
        const AxisJob horizontal_job{
            source_width, destination_width, source_height,
            input_stride, intermediate_stride, 0U,
            horizontal_buffers.half_bandwidth, 0U,
            static_cast<std::uint32_t>(frame_count),
            source_height * input_stride,
            source_height * intermediate_stride, 0U,
        };
        encode_axis(
            command, input, intermediate, horizontal_buffers, horizontal_job,
            @"dsmvc plugin GRAYS horizontal inverse");

        const AxisJob vertical_job{
            source_height, destination_height, destination_width,
            intermediate_stride, output_stride, 1U,
            vertical_buffers.half_bandwidth, 0U,
            static_cast<std::uint32_t>(frame_count),
            source_height * intermediate_stride,
            destination_height * output_stride, 0U,
        };
        encode_axis(
            command, intermediate, output, vertical_buffers, vertical_job,
            @"dsmvc plugin GRAYS vertical inverse");
    }

    void upload(std::span<const MetalFloatFrame> frames,
                MetalFloatStagingStats &stats) {
        const std::size_t row_bytes = static_cast<std::size_t>(source_width)
            * sizeof(float);
        const std::size_t frame_bytes = static_cast<std::size_t>(source_height)
            * input_stride * sizeof(float);
        auto *target_base = static_cast<std::byte *>(input.contents);
        for (std::size_t index = 0U; index < frames.size(); ++index) {
            if (!frames[index].source) {
                throw std::invalid_argument("GRAYS Metal source is null");
            }
            copy_strided_rows(
                target_base + index * frame_bytes,
                static_cast<std::ptrdiff_t>(input_stride) * sizeof(float),
                frames[index].source, frames[index].source_stride_bytes,
                row_bytes, source_height, stats);
        }
    }

    void download(std::span<const MetalFloatFrame> frames,
                  MetalFloatStagingStats &stats) {
        const std::size_t row_bytes = static_cast<std::size_t>(destination_width)
            * sizeof(float);
        const std::size_t frame_bytes = static_cast<std::size_t>(destination_height)
            * output_stride * sizeof(float);
        const auto *source_base = static_cast<const std::byte *>(output.contents);
        for (std::size_t index = 0U; index < frames.size(); ++index) {
            if (!frames[index].destination) {
                throw std::invalid_argument("GRAYS Metal destination is null");
            }
            copy_strided_rows(
                frames[index].destination,
                frames[index].destination_stride_bytes,
                source_base + index * frame_bytes,
                static_cast<std::ptrdiff_t>(output_stride) * sizeof(float),
                row_bytes, destination_height, stats);
        }
    }

    void execute(std::span<const MetalFloatFrame> frames) {
        if (frames.empty() || frames.size() > maximum_batch_size) {
            throw std::invalid_argument("invalid experimental GRAYS Metal batch size");
        }
        const std::scoped_lock lock(execute_mutex);
        @autoreleasepool {
            const os_log_t log = profile_signposts ? profile_log() : nullptr;
            const os_signpost_id_t profile_id = profile_signposts
                ? os_signpost_id_generate(log) : OS_SIGNPOST_ID_INVALID;
            if (profile_signposts) {
                os_signpost_interval_begin(
                    log, profile_id, "DSMVCMetalBatch",
                    "kind=grays frames=%zu", frames.size());
                os_signpost_interval_begin(
                    log, profile_id, "DSMVCMetalUpload");
            }
            MetalFloatStagingStats stats;
            upload(frames, stats);
            if (profile_signposts) {
                os_signpost_interval_end(
                    log, profile_id, "DSMVCMetalUpload",
                    "calls=%zu bytes=%zu", stats.memcpy_calls,
                    stats.copied_bytes);
                os_signpost_interval_begin(
                    log, profile_id, "DSMVCMetalEncode");
            }
            id<MTLCommandBuffer> command = [queue commandBuffer];
            if (command == nil) throw std::runtime_error("Metal command buffer failed");
            command.label = @"dsmvc plugin GRAYS batch";
            encode(command, frames.size());
            if (profile_signposts) {
                os_signpost_interval_end(log, profile_id, "DSMVCMetalEncode");
                os_signpost_interval_begin(log, profile_id, "DSMVCMetalWait");
            }
            [command commit];
            [command waitUntilCompleted];
            if (command.status == MTLCommandBufferStatusError) {
                throw std::runtime_error(
                    ns_error(command.error, "Metal execution failed"));
            }
            if (profile_signposts) {
                os_signpost_interval_end(log, profile_id, "DSMVCMetalWait");
                os_signpost_interval_begin(
                    log, profile_id, "DSMVCMetalDownload");
            }
            download(frames, stats);
            if (profile_signposts) {
                os_signpost_interval_end(
                    log, profile_id, "DSMVCMetalDownload",
                    "calls=%zu bytes=%zu", stats.memcpy_calls,
                    stats.copied_bytes);
                os_signpost_interval_end(log, profile_id, "DSMVCMetalBatch");
            }
            last_memcpy_calls.store(stats.memcpy_calls, std::memory_order_relaxed);
            last_copied_bytes.store(stats.copied_bytes, std::memory_order_relaxed);
        }
    }

    [[nodiscard]] MetalFloatStagingStats staging_stats() const noexcept {
        return {
            last_memcpy_calls.load(std::memory_order_relaxed),
            last_copied_bytes.load(std::memory_order_relaxed),
        };
    }

    std::shared_ptr<const AxisPlan> horizontal;
    std::shared_ptr<const AxisPlan> vertical;
    std::size_t maximum_batch_size = 0U;
    std::size_t threads_per_threadgroup = 0U;
    bool profile_signposts = false;
    std::uint32_t source_width = 0U;
    std::uint32_t source_height = 0U;
    std::uint32_t destination_width = 0U;
    std::uint32_t destination_height = 0U;
    std::uint32_t input_stride = 0U;
    std::uint32_t intermediate_stride = 0U;
    std::uint32_t output_stride = 0U;
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;
    std::array<id<MTLComputePipelineState>, 5> pipelines{};
    PlanBuffers horizontal_buffers;
    PlanBuffers vertical_buffers;
    id<MTLBuffer> input = nil;
    id<MTLBuffer> intermediate = nil;
    id<MTLBuffer> output = nil;
    std::string name;
    std::size_t requested_bytes = 0U;
    std::mutex execute_mutex;
    std::atomic<std::size_t> last_memcpy_calls{0U};
    std::atomic<std::size_t> last_copied_bytes{0U};
};

MetalFloatExecutor::MetalFloatExecutor(
    std::shared_ptr<const AxisPlan> horizontal,
    std::shared_ptr<const AxisPlan> vertical,
    std::size_t maximum_batch_size,
    std::size_t threads_per_threadgroup,
    bool profile_signposts)
    : impl_(std::make_unique<Impl>(
          std::move(horizontal), std::move(vertical), maximum_batch_size,
          threads_per_threadgroup, profile_signposts)) {}

MetalFloatExecutor::~MetalFloatExecutor() = default;

void MetalFloatExecutor::execute(std::span<const MetalFloatFrame> frames) {
    impl_->execute(frames);
}

const std::string &MetalFloatExecutor::device_name() const noexcept {
    return impl_->name;
}

std::size_t MetalFloatExecutor::requested_buffer_bytes() const noexcept {
    return impl_->requested_bytes;
}

MetalFloatStagingStats MetalFloatExecutor::last_staging_stats() const noexcept {
    return impl_->staging_stats();
}

} // namespace dsmvc::experimental
