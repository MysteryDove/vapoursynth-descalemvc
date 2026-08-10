#include "metal_yuv_executor_apple.hpp"

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
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

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

struct ConvertJob {
    std::uint32_t width = 0U;
    std::uint32_t height = 0U;
    std::uint32_t input_stride = 0U;
    std::uint32_t output_stride = 0U;
    std::uint32_t batch_count = 0U;
    std::uint32_t input_frame_stride = 0U;
    std::uint32_t output_frame_stride = 0U;
    std::uint32_t reserved = 0U;
};

static_assert(sizeof(AxisJob) == 48U);
static_assert(sizeof(ConvertJob) == 32U);
static_assert(sizeof(IntegerConversion) == 20U);

struct PlaneGeometry {
    std::uint32_t source_width = 0U;
    std::uint32_t source_height = 0U;
    std::uint32_t destination_width = 0U;
    std::uint32_t destination_height = 0U;
    std::uint32_t input_stride = 0U;
    std::uint32_t intermediate_stride = 0U;
    std::uint32_t result_stride = 0U;
    std::uint32_t output_stride = 0U;
};

struct PlanBuffers {
    id<MTLBuffer> transpose_offsets = nil;
    id<MTLBuffer> transpose_indices = nil;
    id<MTLBuffer> transpose_weights = nil;
    id<MTLBuffer> lower_ld = nil;
    id<MTLBuffer> upper_l = nil;
    id<MTLBuffer> inverse_diagonal = nil;
    std::uint32_t half_bandwidth = 0U;
    std::size_t bytes = 0U;
};

struct PreparedPlane {
    PlaneGeometry geometry;
    PlanBuffers horizontal;
    PlanBuffers vertical;
    id<MTLBuffer> input = nil;
    id<MTLBuffer> intermediate = nil;
    id<MTLBuffer> result = nil;
    id<MTLBuffer> output = nil;
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

[[nodiscard]] std::size_t uploaded_plan_bytes(
    const AxisPlan &plan) noexcept {
    return plan.transpose_offsets.size() * sizeof(std::uint32_t)
        + plan.transpose_indices.size() * sizeof(std::int32_t)
        + plan.transpose_weights.size() * sizeof(float)
        + plan.lower_ld.size() * sizeof(float)
        + plan.upper_l.size() * sizeof(float)
        + plan.inverse_diagonal.size() * sizeof(float);
}

} // namespace

struct MetalYuvExecutor::Impl {
    Impl(std::array<std::shared_ptr<const AxisPlan>, 2> requested_horizontal,
         std::array<std::shared_ptr<const AxisPlan>, 2> requested_vertical,
         std::uint32_t requested_sample_bytes,
         std::size_t requested_maximum_batch_size,
         std::size_t requested_threads_per_threadgroup,
         bool requested_profile_signposts)
        : horizontal(std::move(requested_horizontal)),
          vertical(std::move(requested_vertical)),
          sample_bytes(requested_sample_bytes),
          maximum_batch_size(requested_maximum_batch_size),
          threads_per_threadgroup(requested_threads_per_threadgroup),
          profile_signposts(requested_profile_signposts) {
        if ((sample_bytes != 1U && sample_bytes != 2U)
            || maximum_batch_size == 0U || maximum_batch_size > 64U
            || threads_per_threadgroup == 0U
            || threads_per_threadgroup > 1024U) {
            throw std::invalid_argument("invalid experimental Metal YUV configuration");
        }
        for (std::size_t index = 0U; index < horizontal.size(); ++index) {
            if (!horizontal[index] || !vertical[index]
                || !horizontal[index]->valid() || !vertical[index]->valid()) {
                throw std::invalid_argument("experimental Metal YUV plan is invalid");
            }
            if (horizontal[index]->requires_float64()
                || vertical[index]->requires_float64()) {
                throw std::invalid_argument(
                    "experimental Metal YUV executor supports F32 plans only");
            }
        }

        device = MTLCreateSystemDefaultDevice();
        if (device == nil || !device.hasUnifiedMemory) {
            throw std::runtime_error(
                "experimental Metal YUV executor requires unified memory");
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

        const std::array<NSString *, 5> float_names{
            @"inverse_axis_generic", @"inverse_axis_h1", @"inverse_axis_h3",
            @"inverse_axis_h5", @"inverse_axis_h7"};
        const std::array<NSString *, 5> u8_names{
            @"inverse_axis_u8_generic", @"inverse_axis_u8_h1",
            @"inverse_axis_u8_h3", @"inverse_axis_u8_h5",
            @"inverse_axis_u8_h7"};
        const std::array<NSString *, 5> u16_names{
            @"inverse_axis_u16_generic", @"inverse_axis_u16_h1",
            @"inverse_axis_u16_h3", @"inverse_axis_u16_h5",
            @"inverse_axis_u16_h7"};
        for (std::size_t index = 0U; index < float_names.size(); ++index) {
            float_pipelines[index] = make_pipeline(float_names[index]);
            u8_pipelines[index] = make_pipeline(u8_names[index]);
            u16_pipelines[index] = make_pipeline(u16_names[index]);
        }
        convert_u8 = make_pipeline(@"convert_f32_to_u8");
        convert_u16 = make_pipeline(@"convert_f32_to_u16");
        prepare_planes();
    }

    [[nodiscard]] id<MTLComputePipelineState> make_pipeline(NSString *function_name) {
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
            throw std::length_error("invalid experimental Metal buffer size");
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
        if (values.empty()) throw std::invalid_argument("Metal plan buffer is empty");
        const std::size_t bytes = values.size() * sizeof(Value);
        id<MTLBuffer> buffer = [device newBufferWithBytes:values.data()
                                                  length:bytes
                                                 options:MTLResourceStorageModeShared];
        if (buffer == nil) throw std::runtime_error("Metal plan upload failed");
        buffer.label = label;
        requested_bytes += bytes;
        return buffer;
    }

    [[nodiscard]] PlanBuffers prepare_plan(const AxisPlan &plan) {
        PlanBuffers prepared;
        prepared.transpose_offsets = make_plan_buffer(
            plan.transpose_offsets, @"dsmvc plugin transpose offsets");
        prepared.transpose_indices = make_plan_buffer(
            plan.transpose_indices, @"dsmvc plugin transpose indices");
        prepared.transpose_weights = make_plan_buffer(
            plan.transpose_weights, @"dsmvc plugin transpose weights");
        prepared.lower_ld = make_plan_buffer(
            plan.lower_ld, @"dsmvc plugin lower LD");
        prepared.upper_l = make_plan_buffer(
            plan.upper_l, @"dsmvc plugin upper L");
        prepared.inverse_diagonal = make_plan_buffer(
            plan.inverse_diagonal, @"dsmvc plugin inverse diagonal");
        prepared.half_bandwidth = static_cast<std::uint32_t>(plan.half_bandwidth);
        prepared.bytes = uploaded_plan_bytes(plan);
        return prepared;
    }

    void prepare_planes() {
        for (std::size_t plane_index = 0U; plane_index < planes.size(); ++plane_index) {
            const std::size_t plan_index = plane_index == 0U ? 0U : 1U;
            auto &plane = planes[plane_index];
            plane.geometry.source_width = static_cast<std::uint32_t>(
                horizontal[plan_index]->source_size);
            plane.geometry.destination_width = static_cast<std::uint32_t>(
                horizontal[plan_index]->destination_size);
            plane.geometry.source_height = static_cast<std::uint32_t>(
                vertical[plan_index]->source_size);
            plane.geometry.destination_height = static_cast<std::uint32_t>(
                vertical[plan_index]->destination_size);
            plane.geometry.input_stride = align_up(
                plane.geometry.source_width * sample_bytes, 64U) / sample_bytes;
            plane.geometry.output_stride = align_up(
                plane.geometry.destination_width * sample_bytes, 32U) / sample_bytes;
            plane.geometry.intermediate_stride = align_up(
                plane.geometry.destination_width * sizeof(float), 64U)
                / sizeof(float);
            plane.geometry.result_stride = plane.geometry.intermediate_stride;

            if (plane_index < 2U) {
                plane.horizontal = prepare_plan(*horizontal[plan_index]);
                plane.vertical = prepare_plan(*vertical[plan_index]);
            } else {
                plane.horizontal = planes[1].horizontal;
                plane.vertical = planes[1].vertical;
            }

            const std::size_t input_bytes = static_cast<std::size_t>(
                plane.geometry.source_height) * plane.geometry.input_stride
                * sample_bytes * maximum_batch_size;
            const std::size_t intermediate_bytes = static_cast<std::size_t>(
                plane.geometry.source_height) * plane.geometry.intermediate_stride
                * sizeof(float) * maximum_batch_size;
            const std::size_t result_bytes = static_cast<std::size_t>(
                plane.geometry.destination_height) * plane.geometry.result_stride
                * sizeof(float) * maximum_batch_size;
            const std::size_t output_bytes = static_cast<std::size_t>(
                plane.geometry.destination_height) * plane.geometry.output_stride
                * sample_bytes * maximum_batch_size;
            plane.input = make_empty_buffer(input_bytes, @"dsmvc plugin YUV input");
            if (plane_index == 0U) {
                plane.intermediate = make_empty_buffer(
                    intermediate_bytes, @"dsmvc plugin YUV intermediate");
                plane.result = make_empty_buffer(
                    result_bytes, @"dsmvc plugin YUV result");
            } else {
                if (planes[0].intermediate.length < intermediate_bytes
                    || planes[0].result.length < result_bytes) {
                    throw std::length_error("shared plugin Metal scratch is too small");
                }
                plane.intermediate = planes[0].intermediate;
                plane.result = planes[0].result;
            }
            plane.output = make_empty_buffer(output_bytes, @"dsmvc plugin YUV output");
        }
    }

    [[nodiscard]] static std::size_t pipeline_index(
        std::uint32_t half_bandwidth) noexcept {
        switch (half_bandwidth) {
        case 1U: return 1U;
        case 3U: return 2U;
        case 5U: return 3U;
        case 7U: return 4U;
        default: return 0U;
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

    void dispatch(id<MTLComputeCommandEncoder> encoder,
                  id<MTLComputePipelineState> pipeline,
                  std::size_t count) const {
        const NSUInteger threads = std::min<NSUInteger>(
            static_cast<NSUInteger>(threads_per_threadgroup),
            pipeline.maxTotalThreadsPerThreadgroup);
        [encoder dispatchThreads:MTLSizeMake(count, 1U, 1U)
             threadsPerThreadgroup:MTLSizeMake(threads, 1U, 1U)];
    }

    void upload(std::span<const MetalYuvFrame> frames,
                MetalYuvStagingStats &stats) {
        for (std::size_t plane_index = 0U; plane_index < planes.size(); ++plane_index) {
            const auto &geometry = planes[plane_index].geometry;
            const std::size_t row_bytes = static_cast<std::size_t>(
                geometry.source_width) * sample_bytes;
            const std::size_t frame_bytes = static_cast<std::size_t>(
                geometry.source_height) * geometry.input_stride * sample_bytes;
            auto *target_base = static_cast<std::byte *>(planes[plane_index].input.contents);
            for (std::size_t frame_index = 0U;
                 frame_index < frames.size(); ++frame_index) {
                const auto &frame = frames[frame_index];
                if (!frame.source_planes[plane_index]
                    || frame.source_strides_bytes[plane_index]
                        < static_cast<std::ptrdiff_t>(row_bytes)) {
                    throw std::invalid_argument("invalid source plane for Metal batch");
                }
                const auto *source = static_cast<const std::byte *>(
                    frame.source_planes[plane_index]);
                auto *target = target_base + frame_index * frame_bytes;
                detail::copy_strided_rows(
                    target,
                    static_cast<std::ptrdiff_t>(geometry.input_stride)
                        * sample_bytes,
                    source, frame.source_strides_bytes[plane_index],
                    row_bytes, geometry.source_height, stats);
            }
        }
    }

    void download(std::span<const MetalYuvFrame> frames,
                  MetalYuvStagingStats &stats) {
        for (std::size_t plane_index = 0U; plane_index < planes.size(); ++plane_index) {
            const auto &geometry = planes[plane_index].geometry;
            const std::size_t row_bytes = static_cast<std::size_t>(
                geometry.destination_width) * sample_bytes;
            const std::size_t frame_bytes = static_cast<std::size_t>(
                geometry.destination_height) * geometry.output_stride * sample_bytes;
            const auto *source_base = static_cast<const std::byte *>(
                planes[plane_index].output.contents);
            for (std::size_t frame_index = 0U;
                 frame_index < frames.size(); ++frame_index) {
                const auto &frame = frames[frame_index];
                if (!frame.destination_planes[plane_index]
                    || frame.destination_strides_bytes[plane_index]
                        < static_cast<std::ptrdiff_t>(row_bytes)) {
                    throw std::invalid_argument(
                        "invalid destination plane for Metal batch");
                }
                auto *destination = static_cast<std::byte *>(
                    frame.destination_planes[plane_index]);
                const auto *source = source_base + frame_index * frame_bytes;
                detail::copy_strided_rows(
                    destination,
                    frame.destination_strides_bytes[plane_index],
                    source,
                    static_cast<std::ptrdiff_t>(geometry.output_stride)
                        * sample_bytes,
                    row_bytes, geometry.destination_height, stats);
            }
        }
    }

    void encode_plane(id<MTLCommandBuffer> command, std::size_t plane_index,
                      std::span<const MetalYuvFrame> frames) {
        auto &plane = planes[plane_index];
        const auto &geometry = plane.geometry;
        std::array<IntegerConversion, 64> conversions{};
        for (std::size_t index = 0U; index < frames.size(); ++index) {
            conversions[index] = frames[index].conversions[plane_index];
        }

        const AxisJob horizontal_job{
            geometry.source_width,
            geometry.destination_width,
            geometry.source_height,
            geometry.input_stride,
            geometry.intermediate_stride,
            0U,
            plane.horizontal.half_bandwidth,
            0U,
            static_cast<std::uint32_t>(frames.size()),
            geometry.source_height * geometry.input_stride,
            geometry.source_height * geometry.intermediate_stride,
            0U,
        };
        id<MTLComputeCommandEncoder> horizontal_encoder =
            [command computeCommandEncoder];
        if (horizontal_encoder == nil) throw std::runtime_error("Metal encoder failed");
        horizontal_encoder.label = @"dsmvc plugin YUV horizontal inverse";
        id<MTLComputePipelineState> horizontal_pipeline = sample_bytes == 1U
            ? u8_pipelines[pipeline_index(horizontal_job.half_bandwidth)]
            : u16_pipelines[pipeline_index(horizontal_job.half_bandwidth)];
        [horizontal_encoder setComputePipelineState:horizontal_pipeline];
        [horizontal_encoder setBuffer:plane.input offset:0 atIndex:0];
        [horizontal_encoder setBytes:&horizontal_job
                              length:sizeof(horizontal_job) atIndex:1];
        bind_plan(horizontal_encoder, plane.horizontal);
        [horizontal_encoder setBuffer:plane.intermediate offset:0 atIndex:8];
        [horizontal_encoder setBytes:conversions.data()
                              length:frames.size() * sizeof(conversions.front())
                             atIndex:9];
        dispatch(horizontal_encoder, horizontal_pipeline,
                 static_cast<std::size_t>(geometry.source_height) * frames.size());
        [horizontal_encoder endEncoding];

        const AxisJob vertical_job{
            geometry.source_height,
            geometry.destination_height,
            geometry.destination_width,
            geometry.intermediate_stride,
            geometry.result_stride,
            1U,
            plane.vertical.half_bandwidth,
            0U,
            static_cast<std::uint32_t>(frames.size()),
            geometry.source_height * geometry.intermediate_stride,
            geometry.destination_height * geometry.result_stride,
            0U,
        };
        id<MTLComputeCommandEncoder> vertical_encoder =
            [command computeCommandEncoder];
        if (vertical_encoder == nil) throw std::runtime_error("Metal encoder failed");
        vertical_encoder.label = @"dsmvc plugin YUV vertical inverse";
        id<MTLComputePipelineState> vertical_pipeline =
            float_pipelines[pipeline_index(vertical_job.half_bandwidth)];
        [vertical_encoder setComputePipelineState:vertical_pipeline];
        [vertical_encoder setBuffer:plane.intermediate offset:0 atIndex:0];
        [vertical_encoder setBytes:&vertical_job
                            length:sizeof(vertical_job) atIndex:1];
        bind_plan(vertical_encoder, plane.vertical);
        [vertical_encoder setBuffer:plane.result offset:0 atIndex:8];
        dispatch(vertical_encoder, vertical_pipeline,
                 static_cast<std::size_t>(geometry.destination_width)
                     * frames.size());
        [vertical_encoder endEncoding];

        const ConvertJob convert_job{
            geometry.destination_width,
            geometry.destination_height,
            geometry.result_stride,
            geometry.output_stride,
            static_cast<std::uint32_t>(frames.size()),
            geometry.destination_height * geometry.result_stride,
            geometry.destination_height * geometry.output_stride,
            0U,
        };
        id<MTLComputeCommandEncoder> convert_encoder =
            [command computeCommandEncoder];
        if (convert_encoder == nil) throw std::runtime_error("Metal encoder failed");
        convert_encoder.label = @"dsmvc plugin YUV integer conversion";
        id<MTLComputePipelineState> convert_pipeline =
            sample_bytes == 1U ? convert_u8 : convert_u16;
        [convert_encoder setComputePipelineState:convert_pipeline];
        [convert_encoder setBuffer:plane.result offset:0 atIndex:0];
        [convert_encoder setBytes:&convert_job
                           length:sizeof(convert_job) atIndex:1];
        [convert_encoder setBytes:conversions.data()
                           length:frames.size() * sizeof(conversions.front())
                          atIndex:2];
        [convert_encoder setBuffer:plane.output offset:0 atIndex:3];
        dispatch(convert_encoder, convert_pipeline,
                 static_cast<std::size_t>(geometry.destination_width)
                     * geometry.destination_height * frames.size());
        [convert_encoder endEncoding];
    }

    void execute(std::span<const MetalYuvFrame> frames) {
        if (frames.empty() || frames.size() > maximum_batch_size) {
            throw std::invalid_argument("invalid experimental Metal batch size");
        }
        const std::scoped_lock lock(execute_mutex);
        @autoreleasepool {
            const os_log_t log = profile_signposts ? profile_log() : nullptr;
            const os_signpost_id_t profile_id = profile_signposts
                ? os_signpost_id_generate(log) : OS_SIGNPOST_ID_INVALID;
            if (profile_signposts) {
                os_signpost_interval_begin(
                    log, profile_id, "DSMVCMetalBatch",
                    "frames=%zu", frames.size());
                os_signpost_interval_begin(
                    log, profile_id, "DSMVCMetalUpload");
            }
            MetalYuvStagingStats stats;
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
            command.label = @"dsmvc plugin YUV batch";
            for (std::size_t plane = 0U; plane < planes.size(); ++plane) {
                encode_plane(command, plane, frames);
            }
            if (profile_signposts) {
                os_signpost_interval_end(
                    log, profile_id, "DSMVCMetalEncode");
                os_signpost_interval_begin(
                    log, profile_id, "DSMVCMetalWait");
            }
            [command commit];
            [command waitUntilCompleted];
            if (command.status == MTLCommandBufferStatusError) {
                throw std::runtime_error(
                    ns_error(command.error, "Metal execution failed"));
            }
            if (profile_signposts) {
                os_signpost_interval_end(
                    log, profile_id, "DSMVCMetalWait");
                os_signpost_interval_begin(
                    log, profile_id, "DSMVCMetalDownload");
            }
            download(frames, stats);
            if (profile_signposts) {
                os_signpost_interval_end(
                    log, profile_id, "DSMVCMetalDownload",
                    "calls=%zu bytes=%zu", stats.memcpy_calls,
                    stats.copied_bytes);
                os_signpost_interval_end(
                    log, profile_id, "DSMVCMetalBatch");
            }
            last_memcpy_calls.store(
                stats.memcpy_calls, std::memory_order_relaxed);
            last_copied_bytes.store(
                stats.copied_bytes, std::memory_order_relaxed);
        }
    }

    [[nodiscard]] MetalYuvStagingStats staging_stats() const noexcept {
        return {
            last_memcpy_calls.load(std::memory_order_relaxed),
            last_copied_bytes.load(std::memory_order_relaxed),
        };
    }

    std::array<std::shared_ptr<const AxisPlan>, 2> horizontal;
    std::array<std::shared_ptr<const AxisPlan>, 2> vertical;
    std::uint32_t sample_bytes = 0U;
    std::size_t maximum_batch_size = 0U;
    std::size_t threads_per_threadgroup = 0U;
    bool profile_signposts = false;
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;
    std::array<id<MTLComputePipelineState>, 5> float_pipelines{};
    std::array<id<MTLComputePipelineState>, 5> u8_pipelines{};
    std::array<id<MTLComputePipelineState>, 5> u16_pipelines{};
    id<MTLComputePipelineState> convert_u8 = nil;
    id<MTLComputePipelineState> convert_u16 = nil;
    std::array<PreparedPlane, 3> planes{};
    std::string name;
    std::size_t requested_bytes = 0U;
    std::mutex execute_mutex;
    std::atomic<std::size_t> last_memcpy_calls{0U};
    std::atomic<std::size_t> last_copied_bytes{0U};
};

MetalYuvExecutor::MetalYuvExecutor(
    std::array<std::shared_ptr<const AxisPlan>, 2> horizontal,
    std::array<std::shared_ptr<const AxisPlan>, 2> vertical,
    std::uint32_t sample_bytes, std::size_t maximum_batch_size,
    std::size_t threads_per_threadgroup, bool profile_signposts)
    : impl_(std::make_unique<Impl>(
          std::move(horizontal), std::move(vertical), sample_bytes,
          maximum_batch_size, threads_per_threadgroup,
          profile_signposts)) {}

MetalYuvExecutor::~MetalYuvExecutor() = default;

void MetalYuvExecutor::execute(std::span<const MetalYuvFrame> frames) {
    impl_->execute(frames);
}

const std::string &MetalYuvExecutor::device_name() const noexcept {
    return impl_->name;
}

std::size_t MetalYuvExecutor::requested_buffer_bytes() const noexcept {
    return impl_->requested_bytes;
}

MetalYuvStagingStats MetalYuvExecutor::last_staging_stats() const noexcept {
    return impl_->staging_stats();
}

} // namespace dsmvc::experimental
