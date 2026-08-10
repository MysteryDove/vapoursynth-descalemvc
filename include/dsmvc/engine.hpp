#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace dsmvc {

enum class KernelKind : std::uint8_t {
    bilinear,
    bicubic,
    lanczos,
    spline16,
    spline36,
    spline64,
    custom,
};

enum class BackendKind : std::uint8_t {
    automatic,
    cpu,
    metal,
    vulkan,
    cuda,
};

enum class CpuPath : std::uint8_t {
    // Automatic may fall back to scalar. Explicit SIMD values require that
    // exact implementation to be compiled and supported by the current CPU.
    automatic,
    scalar,
    avx2,
    neon,
};

enum class BorderMode : std::uint8_t {
    zero = 0,
    repeat = 1,
    reflect101 = 2,
    symmetric = 3,
    // Original descale performs one edge-duplicating reflection, not a
    // periodic extension. Kept for the legacy border_handling parameter.
    mirror = 4,
};

enum class F64Mode : std::uint8_t {
    automatic = 0,
    float32_only = 1,
    float64_only = 2,
};

struct KernelSpec {
    KernelKind kind = KernelKind::bicubic;
    std::int32_t taps = 3;
    double b = 0.0;
    double c = 0.5;
};

struct AxisRequest {
    std::int32_t source_size = 0;
    std::int32_t destination_size = 0;
    double active_length = 0.0;
    double shift = 0.0;
    KernelSpec kernel{};
    BorderMode border = BorderMode::symmetric;
    F64Mode f64_mode = F64Mode::automatic;
};

using CustomKernel = std::function<double(double)>;

// Immutable inverse-only planner output. The CPU descale executor does not
// need a forward projection table, so plans retain only inverse coefficients.
struct AxisPlan {
    std::int32_t source_size = 0;
    std::int32_t destination_size = 0;
    std::int32_t support = 0;
    std::int32_t half_bandwidth = 0;
    double active_length = 0.0;
    double shift = 0.0;

    std::vector<std::uint32_t> transpose_offsets;
    std::vector<std::int32_t> transpose_indices;
    std::vector<float> transpose_weights;
    std::vector<float> lower_ld;
    std::vector<float> upper_l;
    std::vector<float> inverse_diagonal;

    // The Float32 normal-equation path is retained for well-conditioned axes.
    // This is either a conservative lower bound or a Hager 1-norm estimate.
    // Unsafe axes also retain the original Float64 transpose, normal matrix,
    // and LDLT bands. The unfactored normal matrix supports independent
    // high-precision residual evaluation without reconstructing it from F32.
    double normal_rcond = 1.0;
    double normal_inf_norm = 0.0;
    std::vector<double> transpose_weights_f64;
    std::vector<double> normal_bands_f64;
    std::vector<double> ldlt_bands_f64;
    std::vector<double> inverse_diagonal_f64;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] bool requires_float64() const noexcept;
    [[nodiscard]] std::size_t storage_bytes() const noexcept;
};

struct PlannerCacheStats {
    std::uint64_t plan_hits = 0;
    std::uint64_t plan_builds = 0;
    std::uint64_t geometry_hits = 0;
    std::uint64_t geometry_builds = 0;
    std::size_t plan_entries = 0;
    std::size_t plan_resident_bytes = 0;
    std::size_t geometry_entries = 0;
    std::size_t geometry_resident_bytes = 0;
};

struct CpuPlanPackingStats {
    std::uint64_t pack_executions = 0;
    std::uint64_t single_flight_waits = 0;
    std::uint64_t single_flight_wait_nanoseconds = 0;
    std::uint64_t lazy_requests = 0;
    std::uint64_t lazy_hits = 0;
    std::uint64_t maximum_concurrent_packs = 0;
};

struct BackendCapability {
    BackendKind kind{};
    const char *name = "";
    bool compiled = false;
    bool device_available = false;
};

struct IntegerConversion {
    float input_offset = 0.0F;
    float input_scale = 1.0F;
    float output_scale = 1.0F;
    float output_offset = 0.0F;
    std::uint32_t output_maximum = 255U;
};

[[nodiscard]] AxisPlan build_axis_plan(
    const AxisRequest &request, const CustomKernel &custom_kernel = {});
[[nodiscard]] std::shared_ptr<const AxisPlan> get_or_build_axis_plan(
    const AxisRequest &request, const CustomKernel &custom_kernel = {});
[[nodiscard]] PlannerCacheStats planner_cache_stats();
void clear_planner_caches();

void inverse_axis_f32(const AxisPlan &plan,
                      const float *input, std::ptrdiff_t input_stride,
                      float *output, std::ptrdiff_t output_stride);

inline void inverse_axis_f32(const AxisPlan &plan,
                             std::span<const float> input,
                             std::span<float> output) {
    if (input.size() < static_cast<std::size_t>(plan.source_size)
        || output.size() < static_cast<std::size_t>(plan.destination_size)) {
        throw std::invalid_argument("inverse axis spans are too small");
    }
    inverse_axis_f32(plan, input.data(), 1, output.data(), 1);
}

[[nodiscard]] BackendKind parse_backend(std::string_view name);
[[nodiscard]] BackendKind resolve_backend(BackendKind requested);
[[nodiscard]] std::vector<BackendCapability> backend_capabilities();
[[nodiscard]] const char *backend_name(BackendKind kind) noexcept;

[[nodiscard]] bool cpu_avx2_compiled() noexcept;
[[nodiscard]] bool cpu_avx2_available() noexcept;
[[nodiscard]] bool cpu_neon_compiled() noexcept;
[[nodiscard]] bool cpu_neon_available() noexcept;
[[nodiscard]] bool metal_compiled() noexcept;
[[nodiscard]] bool metal_available() noexcept;
[[nodiscard]] bool vulkan_compiled() noexcept;
[[nodiscard]] bool vulkan_available() noexcept;
[[nodiscard]] bool cuda_compiled() noexcept;
[[nodiscard]] bool cuda_available() noexcept;

class CpuExecutor {
public:
    explicit CpuExecutor(CpuPath requested = CpuPath::automatic);
    ~CpuExecutor();
    CpuExecutor(const CpuExecutor &) noexcept = default;
    CpuExecutor &operator=(const CpuExecutor &) noexcept = default;
    CpuExecutor(CpuExecutor &&) noexcept = default;
    CpuExecutor &operator=(CpuExecutor &&) noexcept = default;

    [[nodiscard]] CpuPath path() const noexcept;
    [[nodiscard]] const char *name() const noexcept;
    [[nodiscard]] CpuPlanPackingStats packing_stats() const noexcept;

    void prepare(std::shared_ptr<const AxisPlan> plan) const;
    void defer(std::shared_ptr<const AxisPlan> plan) const;
    void seal() const;

    // Buffer contract for this class and Executor below: strides cover the
    // logical width of each row. Callers need only allocate (row_count - 1) *
    // row_stride + logical_width elements; the final row does not require
    // pitch padding. Executors do not access or modify row padding outside the
    // logical width.

    void inverse_rows(const AxisPlan &plan,
                      const float *input, std::ptrdiff_t input_row_stride,
                      float *output, std::ptrdiff_t output_row_stride,
                      std::int32_t row_count) const;

    void inverse_columns(const AxisPlan &plan,
                         const float *input, std::ptrdiff_t input_row_stride,
                         float *output, std::ptrdiff_t output_row_stride,
                         std::int32_t column_count) const;

    void inverse_2d(const AxisPlan &horizontal, const AxisPlan &vertical,
                    const float *input, std::ptrdiff_t input_row_stride,
                    float *output, std::ptrdiff_t output_row_stride) const;

    void inverse_2d_u8(
        const AxisPlan &horizontal, const AxisPlan &vertical,
        const std::uint8_t *input, std::ptrdiff_t input_row_stride,
        std::uint8_t *output, std::ptrdiff_t output_row_stride,
        const IntegerConversion &conversion) const;

    void inverse_2d_u16(
        const AxisPlan &horizontal, const AxisPlan &vertical,
        const std::uint16_t *input, std::ptrdiff_t input_row_stride,
        std::uint16_t *output, std::ptrdiff_t output_row_stride,
        const IntegerConversion &conversion) const;

    void inverse_2d_u8_streamed(
        const AxisPlan &horizontal, const AxisPlan &vertical,
        const std::uint8_t *input, std::ptrdiff_t input_row_stride,
        std::uint8_t *output, std::ptrdiff_t output_row_stride,
        const IntegerConversion &conversion) const;

    void inverse_2d_u16_streamed(
        const AxisPlan &horizontal, const AxisPlan &vertical,
        const std::uint16_t *input, std::ptrdiff_t input_row_stride,
        std::uint16_t *output, std::ptrdiff_t output_row_stride,
        const IntegerConversion &conversion) const;

private:
    struct Impl;
    CpuPath path_ = CpuPath::scalar;
    std::shared_ptr<Impl> impl_;

    template <class Sample>
    void inverse_2d_integer(
        const AxisPlan &horizontal, const AxisPlan &vertical,
        const Sample *input, std::ptrdiff_t input_row_stride,
        Sample *output, std::ptrdiff_t output_row_stride,
        const IntegerConversion &conversion) const;

    template <class Sample>
    void inverse_2d_integer_streamed(
        const AxisPlan &horizontal, const AxisPlan &vertical,
        const Sample *input, std::ptrdiff_t input_row_stride,
        Sample *output, std::ptrdiff_t output_row_stride,
        const IntegerConversion &conversion) const;
};

// Backend-neutral executor used by the plugin. Automatic dispatch deliberately
// remains CPU-first; callers opt in to a GPU backend explicitly.
class Executor {
public:
    explicit Executor(
        BackendKind requested = BackendKind::automatic,
        CpuPath cpu_path = CpuPath::automatic);
    ~Executor();
    Executor(const Executor &) noexcept = default;
    Executor &operator=(const Executor &) noexcept = default;
    Executor(Executor &&) noexcept = default;
    Executor &operator=(Executor &&) noexcept = default;

    [[nodiscard]] BackendKind backend() const noexcept;
    [[nodiscard]] const char *name() const noexcept;
    [[nodiscard]] bool input_cache_enabled() const noexcept;
    [[nodiscard]] CpuPlanPackingStats cpu_plan_packing_stats() const noexcept;

    void prepare(std::shared_ptr<const AxisPlan> plan) const;
    void defer(std::shared_ptr<const AxisPlan> plan) const;
    void seal() const;

    void inverse_rows(const AxisPlan &plan,
                      const float *input, std::ptrdiff_t input_row_stride,
                      float *output, std::ptrdiff_t output_row_stride,
                      std::int32_t row_count,
                      std::shared_ptr<const void> input_lifetime = {}) const;

    void inverse_columns(const AxisPlan &plan,
                         const float *input, std::ptrdiff_t input_row_stride,
                         float *output, std::ptrdiff_t output_row_stride,
                         std::int32_t column_count,
                         std::shared_ptr<const void> input_lifetime = {}) const;

    void inverse_2d(const AxisPlan &horizontal, const AxisPlan &vertical,
                    const float *input, std::ptrdiff_t input_row_stride,
                    float *output, std::ptrdiff_t output_row_stride,
                    std::shared_ptr<const void> input_lifetime = {}) const;

    void inverse_2d_u8(
        const AxisPlan &horizontal, const AxisPlan &vertical,
        const std::uint8_t *input, std::ptrdiff_t input_row_stride,
        std::uint8_t *output, std::ptrdiff_t output_row_stride,
        const IntegerConversion &conversion,
        std::shared_ptr<const void> input_lifetime = {}) const;

    void inverse_2d_u16(
        const AxisPlan &horizontal, const AxisPlan &vertical,
        const std::uint16_t *input, std::ptrdiff_t input_row_stride,
        std::uint16_t *output, std::ptrdiff_t output_row_stride,
        const IntegerConversion &conversion,
        std::shared_ptr<const void> input_lifetime = {}) const;

    void inverse_2d_u8_streamed(
        const AxisPlan &horizontal, const AxisPlan &vertical,
        const std::uint8_t *input, std::ptrdiff_t input_row_stride,
        std::uint8_t *output, std::ptrdiff_t output_row_stride,
        const IntegerConversion &conversion,
        std::shared_ptr<const void> input_lifetime = {}) const;

    void inverse_2d_u16_streamed(
        const AxisPlan &horizontal, const AxisPlan &vertical,
        const std::uint16_t *input, std::ptrdiff_t input_row_stride,
        std::uint16_t *output, std::ptrdiff_t output_row_stride,
        const IntegerConversion &conversion,
        std::shared_ptr<const void> input_lifetime = {}) const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace dsmvc
