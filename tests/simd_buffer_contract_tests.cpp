#include "checked_size.hpp"

#include <dsmvc/engine.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {

void require(bool condition, std::string message) {
    if (!condition) throw std::runtime_error(std::move(message));
}

template <class Function>
void expect_invalid(Function &&function, std::string_view label) {
    try {
        std::forward<Function>(function)();
    } catch (const std::invalid_argument &) {
        return;
    }
    throw std::runtime_error(std::string{label}
                             + " did not throw std::invalid_argument");
}

template <class Function>
void expect_length_error(Function &&function, std::string_view label) {
    try {
        std::forward<Function>(function)();
    } catch (const std::length_error &) {
        return;
    }
    throw std::runtime_error(std::string{label}
                             + " did not throw std::length_error");
}

[[nodiscard]] std::size_t system_page_size() {
#if defined(_WIN32)
    SYSTEM_INFO information{};
    GetSystemInfo(&information);
    return static_cast<std::size_t>(information.dwPageSize);
#else
    const long value = sysconf(_SC_PAGESIZE);
    if (value <= 0) throw std::runtime_error("could not query system page size");
    return static_cast<std::size_t>(value);
#endif
}

template <class Value>
class GuardPageBuffer final {
public:
    explicit GuardPageBuffer(std::size_t count) : count_(count) {
        if (count == 0U) throw std::invalid_argument("guard buffer is empty");
        const std::size_t bytes = dsmvc::detail::checked_size_product(
            count, sizeof(Value), "guard buffer");
        const std::size_t page = system_page_size();
        accessible_bytes_ = dsmvc::detail::checked_size_round_up(
            bytes, page, "guard buffer mapping");
        const std::size_t mapping_bytes = dsmvc::detail::checked_size_add(
            accessible_bytes_, page, "guard buffer mapping");
#if defined(_WIN32)
        mapping_ = VirtualAlloc(
            nullptr, mapping_bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        if (mapping_ == nullptr) {
            throw std::runtime_error("VirtualAlloc failed for guard buffer");
        }
        DWORD old_protection = 0;
        auto *guard = static_cast<std::byte *>(mapping_) + accessible_bytes_;
        if (VirtualProtect(guard, page, PAGE_NOACCESS, &old_protection) == 0) {
            VirtualFree(mapping_, 0, MEM_RELEASE);
            mapping_ = nullptr;
            throw std::runtime_error("VirtualProtect failed for guard buffer");
        }
#else
        mapping_ = mmap(
            nullptr, mapping_bytes, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapping_ == MAP_FAILED) {
            mapping_ = nullptr;
            throw std::runtime_error("mmap failed for guard buffer");
        }
        auto *guard = static_cast<std::byte *>(mapping_) + accessible_bytes_;
        if (mprotect(guard, page, PROT_NONE) != 0) {
            munmap(mapping_, mapping_bytes);
            mapping_ = nullptr;
            throw std::runtime_error("mprotect failed for guard buffer");
        }
#endif
        data_ = reinterpret_cast<Value *>(
            static_cast<std::byte *>(mapping_) + accessible_bytes_ - bytes);
    }

    ~GuardPageBuffer() {
        if (mapping_ == nullptr) return;
#if defined(_WIN32)
        VirtualFree(mapping_, 0, MEM_RELEASE);
#else
        const std::size_t mapping_bytes = accessible_bytes_ + system_page_size();
        munmap(mapping_, mapping_bytes);
#endif
    }

    GuardPageBuffer(const GuardPageBuffer &) = delete;
    GuardPageBuffer &operator=(const GuardPageBuffer &) = delete;

    [[nodiscard]] Value *data() noexcept { return data_; }
    [[nodiscard]] const Value *data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return count_; }

private:
    void *mapping_ = nullptr;
    Value *data_ = nullptr;
    std::size_t count_ = 0U;
    std::size_t accessible_bytes_ = 0U;
};

[[nodiscard]] dsmvc::AxisPlan make_plan(
    std::int32_t source, std::int32_t destination, double shift = 0.125) {
    dsmvc::AxisRequest request;
    request.source_size = source;
    request.destination_size = destination;
    request.active_length = static_cast<double>(destination) - 0.25;
    request.shift = shift;
    request.kernel.kind = dsmvc::KernelKind::bicubic;
    request.kernel.b = 0.0;
    request.kernel.c = 0.5;
    request.border = dsmvc::BorderMode::symmetric;
    request.f64_mode = dsmvc::F64Mode::float32_only;
    return dsmvc::build_axis_plan(request);
}

[[nodiscard]] std::size_t matrix_elements(
    std::int32_t rows, std::ptrdiff_t stride, std::int32_t logical_width) {
    return static_cast<std::size_t>(rows - 1)
            * static_cast<std::size_t>(stride)
        + static_cast<std::size_t>(logical_width);
}

template <class Value>
[[nodiscard]] Value output_sentinel();

template <>
[[nodiscard]] float output_sentinel<float>() { return -1234.5F; }

template <>
[[nodiscard]] std::uint8_t output_sentinel<std::uint8_t>() { return 0xa5U; }

template <>
[[nodiscard]] std::uint16_t output_sentinel<std::uint16_t>() { return 0xa55aU; }

template <class Value>
void fill_input(Value *data, std::int32_t rows, std::ptrdiff_t stride,
                std::int32_t width, std::uint32_t maximum) {
    std::uint32_t state = 0x81a5c3e7U + static_cast<std::uint32_t>(width);
    for (std::int32_t row = 0; row < rows; ++row) {
        for (std::int32_t column = 0; column < width; ++column) {
            state ^= state << 13U;
            state ^= state >> 17U;
            state ^= state << 5U;
            if constexpr (std::is_same_v<Value, float>) {
                data[static_cast<std::ptrdiff_t>(row) * stride + column] =
                    static_cast<float>(state & 0xffffU) / 65535.0F;
            } else {
                data[static_cast<std::ptrdiff_t>(row) * stride + column] =
                    static_cast<Value>(state % (maximum + 1U));
            }
        }
    }
}

template <class Value>
void require_padding_unchanged(
    const Value *data, std::int32_t rows, std::ptrdiff_t stride,
    std::int32_t logical_width, Value sentinel, std::string_view label) {
    for (std::int32_t row = 0; row + 1 < rows; ++row) {
        for (std::ptrdiff_t column = logical_width; column < stride; ++column) {
            require(data[static_cast<std::ptrdiff_t>(row) * stride + column]
                        == sentinel,
                    std::string{label} + " modified public row padding");
        }
    }
}

void require_float_agreement(
    const float *reference, const float *candidate, std::int32_t rows,
    std::int32_t columns, std::ptrdiff_t stride, std::string_view label) {
    double maximum = 0.0;
    for (std::int32_t row = 0; row < rows; ++row) {
        for (std::int32_t column = 0; column < columns; ++column) {
            const auto index = static_cast<std::ptrdiff_t>(row) * stride + column;
            require(std::isfinite(candidate[index]),
                    std::string{label} + " produced a non-finite value");
            maximum = std::max(
                maximum, std::abs(static_cast<double>(reference[index])
                                  - static_cast<double>(candidate[index])));
        }
    }
    require(maximum <= 3.0e-5,
            std::string{label} + " differs from CPU scalar");
}

template <class Value>
void require_integer_agreement(
    const Value *reference, const Value *candidate, std::int32_t rows,
    std::int32_t columns, std::ptrdiff_t stride, std::string_view label) {
    std::uint32_t maximum = 0U;
    for (std::int32_t row = 0; row < rows; ++row) {
        for (std::int32_t column = 0; column < columns; ++column) {
            const auto index = static_cast<std::ptrdiff_t>(row) * stride + column;
            const auto left = static_cast<std::uint32_t>(reference[index]);
            const auto right = static_cast<std::uint32_t>(candidate[index]);
            maximum = std::max(maximum, left > right ? left - right : right - left);
        }
    }
    require(maximum <= 1U,
            std::string{label} + " differs from CPU scalar by more than one code");
}

void test_rows_and_columns(
    dsmvc::CpuPath path, std::int32_t lanes, std::int32_t residue) {
    const auto plan = make_plan(2 * lanes + residue, lanes + residue);
    const dsmvc::CpuExecutor scalar(dsmvc::CpuPath::scalar);
    const dsmvc::CpuExecutor executor(path);

    const std::int32_t row_count = lanes + 3;
    const std::ptrdiff_t input_stride = plan.source_size + 3;
    const std::ptrdiff_t output_stride = plan.destination_size + 5;
    GuardPageBuffer<float> row_input(matrix_elements(
        row_count, input_stride, plan.source_size));
    GuardPageBuffer<float> row_output(matrix_elements(
        row_count, output_stride, plan.destination_size));
    std::fill_n(row_input.data(), row_input.size(), -77.0F);
    std::fill_n(row_output.data(), row_output.size(), output_sentinel<float>());
    fill_input(row_input.data(), row_count, input_stride, plan.source_size, 0U);
    std::vector<float> row_reference(row_output.size(), output_sentinel<float>());
    scalar.inverse_rows(
        plan, row_input.data(), input_stride,
        row_reference.data(), output_stride, row_count);
    executor.inverse_rows(
        plan, row_input.data(), input_stride,
        row_output.data(), output_stride, row_count);
    require_float_agreement(
        row_reference.data(), row_output.data(), row_count,
        plan.destination_size, output_stride, "guarded rows");
    require_padding_unchanged(
        row_output.data(), row_count, output_stride, plan.destination_size,
        output_sentinel<float>(), "guarded rows");

    const std::int32_t columns = lanes + residue;
    const std::ptrdiff_t column_stride = columns + 3;
    GuardPageBuffer<float> column_input(matrix_elements(
        plan.source_size, column_stride, columns));
    GuardPageBuffer<float> column_output(matrix_elements(
        plan.destination_size, column_stride, columns));
    std::fill_n(column_input.data(), column_input.size(), -66.0F);
    std::fill_n(
        column_output.data(), column_output.size(), output_sentinel<float>());
    fill_input(
        column_input.data(), plan.source_size, column_stride, columns, 0U);
    std::vector<float> column_reference(
        column_output.size(), output_sentinel<float>());
    scalar.inverse_columns(
        plan, column_input.data(), column_stride,
        column_reference.data(), column_stride, columns);
    executor.inverse_columns(
        plan, column_input.data(), column_stride,
        column_output.data(), column_stride, columns);
    require_float_agreement(
        column_reference.data(), column_output.data(), plan.destination_size,
        columns, column_stride, "guarded columns");
    require_padding_unchanged(
        column_output.data(), plan.destination_size, column_stride, columns,
        output_sentinel<float>(), "guarded columns");
}

void test_float_2d(
    dsmvc::CpuPath path, std::int32_t lanes, std::int32_t residue) {
    const auto horizontal = make_plan(2 * lanes + residue, lanes + residue);
    const auto vertical = make_plan(2 * lanes + 3, lanes + 1, -0.25);
    const std::ptrdiff_t input_stride = horizontal.source_size + 3;
    const std::ptrdiff_t output_stride = horizontal.destination_size + 5;
    GuardPageBuffer<float> input(matrix_elements(
        vertical.source_size, input_stride, horizontal.source_size));
    GuardPageBuffer<float> output(matrix_elements(
        vertical.destination_size, output_stride, horizontal.destination_size));
    std::fill_n(input.data(), input.size(), -55.0F);
    std::fill_n(output.data(), output.size(), output_sentinel<float>());
    fill_input(
        input.data(), vertical.source_size, input_stride,
        horizontal.source_size, 0U);
    std::vector<float> reference(output.size(), output_sentinel<float>());
    const dsmvc::CpuExecutor scalar(dsmvc::CpuPath::scalar);
    scalar.inverse_2d(
        horizontal, vertical, input.data(), input_stride,
        reference.data(), output_stride);
    const dsmvc::CpuExecutor executor(path);
    executor.inverse_2d(
        horizontal, vertical, input.data(), input_stride,
        output.data(), output_stride);
    require_float_agreement(
        reference.data(), output.data(), vertical.destination_size,
        horizontal.destination_size, output_stride, "guarded F32 2D");
    require_padding_unchanged(
        output.data(), vertical.destination_size, output_stride,
        horizontal.destination_size, output_sentinel<float>(),
        "guarded F32 2D");
}

template <class Sample>
void test_integer_2d(
    dsmvc::CpuPath path, std::int32_t lanes, std::int32_t residue,
    const dsmvc::IntegerConversion &conversion, std::uint32_t input_maximum) {
    const auto horizontal = make_plan(2 * lanes + residue, lanes + residue);
    const auto vertical = make_plan(2 * lanes + 3, lanes + 1, -0.25);
    const std::ptrdiff_t input_stride = horizontal.source_size + 3;
    const std::ptrdiff_t output_stride = horizontal.destination_size + 5;
    GuardPageBuffer<Sample> input(matrix_elements(
        vertical.source_size, input_stride, horizontal.source_size));
    GuardPageBuffer<Sample> buffered(matrix_elements(
        vertical.destination_size, output_stride, horizontal.destination_size));
    GuardPageBuffer<Sample> streamed(matrix_elements(
        vertical.destination_size, output_stride, horizontal.destination_size));
    std::fill_n(input.data(), input.size(), static_cast<Sample>(3));
    std::fill_n(buffered.data(), buffered.size(), output_sentinel<Sample>());
    std::fill_n(streamed.data(), streamed.size(), output_sentinel<Sample>());
    fill_input(
        input.data(), vertical.source_size, input_stride,
        horizontal.source_size, input_maximum);
    std::vector<Sample> reference(buffered.size(), output_sentinel<Sample>());
    const dsmvc::CpuExecutor scalar(dsmvc::CpuPath::scalar);
    if constexpr (std::is_same_v<Sample, std::uint8_t>) {
        scalar.inverse_2d_u8(
            horizontal, vertical, input.data(), input_stride,
            reference.data(), output_stride, conversion);
    } else {
        scalar.inverse_2d_u16(
            horizontal, vertical, input.data(), input_stride,
            reference.data(), output_stride, conversion);
    }

    const dsmvc::CpuExecutor executor(path);
    if constexpr (std::is_same_v<Sample, std::uint8_t>) {
        executor.inverse_2d_u8(
            horizontal, vertical, input.data(), input_stride,
            buffered.data(), output_stride, conversion);
        executor.inverse_2d_u8_streamed(
            horizontal, vertical, input.data(), input_stride,
            streamed.data(), output_stride, conversion);
    } else {
        executor.inverse_2d_u16(
            horizontal, vertical, input.data(), input_stride,
            buffered.data(), output_stride, conversion);
        executor.inverse_2d_u16_streamed(
            horizontal, vertical, input.data(), input_stride,
            streamed.data(), output_stride, conversion);
    }
    require_integer_agreement(
        reference.data(), buffered.data(), vertical.destination_size,
        horizontal.destination_size, output_stride, "guarded integer buffered");
    require_integer_agreement(
        reference.data(), streamed.data(), vertical.destination_size,
        horizontal.destination_size, output_stride, "guarded integer streamed");
    require_padding_unchanged(
        buffered.data(), vertical.destination_size, output_stride,
        horizontal.destination_size, output_sentinel<Sample>(),
        "guarded integer buffered");
    require_padding_unchanged(
        streamed.data(), vertical.destination_size, output_stride,
        horizontal.destination_size, output_sentinel<Sample>(),
        "guarded integer streamed");
}

template <class ExecutorType>
void test_short_strides(ExecutorType &executor, std::string_view label) {
    const auto horizontal = make_plan(17, 11);
    const auto vertical = make_plan(13, 9, -0.25);
    constexpr std::int32_t vectors = 9;
    std::vector<float> float_input(17U * 16000U, 0.25F);
    std::vector<float> float_output(11U * 16000U, 0.0F);
    std::vector<std::uint8_t> u8_input(17U * 13U, 127U);
    std::vector<std::uint8_t> u8_output(11U * 9U, 0U);
    std::vector<std::uint16_t> u16_input(17U * 13U, 32767U);
    std::vector<std::uint16_t> u16_output(11U * 9U, 0U);
    const dsmvc::IntegerConversion u8_conversion{
        0.0F, 1.0F / 255.0F, 255.0F, 0.0F, 255U};
    const dsmvc::IntegerConversion u16_conversion{
        0.0F, 1.0F / 65535.0F, 65535.0F, 0.0F, 65535U};

    expect_invalid([&] {
        executor.inverse_rows(
            horizontal, float_input.data(), horizontal.source_size - 1,
            float_output.data(), horizontal.destination_size, 1);
    }, std::string{label} + " rows input stride");
    expect_invalid([&] {
        executor.inverse_rows(
            horizontal, float_input.data(), horizontal.source_size,
            float_output.data(), horizontal.destination_size - 1, 1);
    }, std::string{label} + " rows output stride");
    expect_invalid([&] {
        executor.inverse_columns(
            vertical, float_input.data(), vectors - 1,
            float_output.data(), vectors, vectors);
    }, std::string{label} + " columns input stride");
    expect_invalid([&] {
        executor.inverse_columns(
            vertical, float_input.data(), vectors,
            float_output.data(), vectors - 1, vectors);
    }, std::string{label} + " columns output stride");
    expect_invalid([&] {
        executor.inverse_2d(
            horizontal, vertical, float_input.data(), horizontal.source_size - 1,
            float_output.data(), horizontal.destination_size);
    }, std::string{label} + " F32 2D input stride");
    expect_invalid([&] {
        executor.inverse_2d(
            horizontal, vertical, float_input.data(), horizontal.source_size,
            float_output.data(), horizontal.destination_size - 1);
    }, std::string{label} + " F32 2D output stride");

    expect_invalid([&] {
        executor.inverse_2d_u8(
            horizontal, vertical, u8_input.data(), horizontal.source_size - 1,
            u8_output.data(), horizontal.destination_size, u8_conversion);
    }, std::string{label} + " U8 buffered input stride");
    expect_invalid([&] {
        executor.inverse_2d_u8_streamed(
            horizontal, vertical, u8_input.data(), horizontal.source_size,
            u8_output.data(), horizontal.destination_size - 1, u8_conversion);
    }, std::string{label} + " U8 streamed output stride");
    expect_invalid([&] {
        executor.inverse_2d_u16(
            horizontal, vertical, u16_input.data(), horizontal.source_size - 1,
            u16_output.data(), horizontal.destination_size, u16_conversion);
    }, std::string{label} + " U16 buffered input stride");
    expect_invalid([&] {
        executor.inverse_2d_u16_streamed(
            horizontal, vertical, u16_input.data(), horizontal.source_size,
            u16_output.data(), horizontal.destination_size - 1, u16_conversion);
    }, std::string{label} + " U16 streamed output stride");

    expect_invalid([&] {
        executor.inverse_rows(
            horizontal, float_input.data(), horizontal.source_size - 1,
            float_output.data(), horizontal.destination_size, 16000);
    }, std::string{label} + " parallel rows pre-dispatch stride");

    constexpr std::int32_t large_columns = 32768;
    std::vector<float> large_column_input(
        static_cast<std::size_t>(vertical.source_size) * large_columns, 0.25F);
    std::vector<float> large_column_output(
        static_cast<std::size_t>(vertical.destination_size) * large_columns,
        0.0F);
    expect_invalid([&] {
        executor.inverse_columns(
            vertical, large_column_input.data(), large_columns - 1,
            large_column_output.data(), large_columns, large_columns);
    }, std::string{label} + " parallel columns pre-dispatch stride");
}

void test_cpu_paths_and_strides() {
    dsmvc::CpuExecutor scalar(dsmvc::CpuPath::scalar);
    test_short_strides(scalar, "scalar");

    if (dsmvc::cpu_avx2_available()) {
        dsmvc::CpuExecutor avx2(dsmvc::CpuPath::avx2);
        test_short_strides(avx2, "AVX2");
        for (std::int32_t residue = 1; residue < 8; ++residue) {
            test_rows_and_columns(dsmvc::CpuPath::avx2, 8, residue);
            test_float_2d(dsmvc::CpuPath::avx2, 8, residue);
            test_integer_2d<std::uint8_t>(
                dsmvc::CpuPath::avx2, 8, residue,
                {0.0F, 1.0F / 255.0F, 255.0F, 0.0F, 255U}, 255U);
            test_integer_2d<std::uint16_t>(
                dsmvc::CpuPath::avx2, 8, residue,
                {0.0F, 1.0F / 65535.0F, 65535.0F, 0.0F, 65535U},
                65535U);
        }
    }
    if (dsmvc::cpu_neon_available()) {
        dsmvc::CpuExecutor neon(dsmvc::CpuPath::neon);
        test_short_strides(neon, "NEON");
        for (std::int32_t residue = 1; residue < 4; ++residue) {
            test_rows_and_columns(dsmvc::CpuPath::neon, 4, residue);
            test_float_2d(dsmvc::CpuPath::neon, 4, residue);
            test_integer_2d<std::uint8_t>(
                dsmvc::CpuPath::neon, 4, residue,
                {0.0F, 1.0F / 255.0F, 255.0F, 0.0F, 255U}, 255U);
            test_integer_2d<std::uint16_t>(
                dsmvc::CpuPath::neon, 4, residue,
                {0.0F, 1.0F / 65535.0F, 65535.0F, 0.0F, 65535U},
                65535U);
        }
    }
}

void test_accelerator_short_strides() {
    for (const auto capability : dsmvc::backend_capabilities()) {
        if ((capability.kind != dsmvc::BackendKind::cuda
             && capability.kind != dsmvc::BackendKind::vulkan)
            || !capability.compiled || !capability.device_available) {
            continue;
        }
        dsmvc::Executor executor(capability.kind);
        test_short_strides(executor, capability.name);
    }
}

void test_checked_extreme_plan() {
    expect_length_error([] {
        (void)dsmvc::detail::checked_size_round_up(
            std::numeric_limits<std::size_t>::max(), 8U, "test round-up");
    }, "checked size round-up");
    expect_length_error([] {
        (void)dsmvc::detail::checked_size_product(
            std::numeric_limits<std::size_t>::max(), 2U, "test product");
    }, "checked size product");

    auto extreme = std::make_shared<dsmvc::AxisPlan>();
    extreme->source_size = std::numeric_limits<std::int32_t>::max();
    extreme->destination_size = 1;
    extreme->support = 1;
    extreme->half_bandwidth = 0;
    extreme->active_length = 1.0;
    extreme->shift = 0.0;
    extreme->transpose_offsets = {0U, 1U};
    extreme->transpose_indices = {0};
    extreme->transpose_weights = {1.0F};
    extreme->inverse_diagonal = {1.0F};
    extreme->normal_rcond = 1.0;
    extreme->normal_inf_norm = 0.0;
    require(extreme->valid(), "extreme CPU packing fixture is invalid");
    dsmvc::CpuExecutor executor(dsmvc::CpuPath::scalar);
    expect_length_error([&] { executor.prepare(extreme); },
                        "extreme CPU plan packing");
}

void test_strict_cpu_path_selection() {
    if (!dsmvc::cpu_avx2_available()) {
        try {
            dsmvc::CpuExecutor executor(dsmvc::CpuPath::avx2);
            (void)executor;
        } catch (const std::runtime_error &) {
            goto avx2_rejected;
        }
        throw std::runtime_error("unavailable explicit AVX2 path did not fail");
    }
avx2_rejected:
    if (!dsmvc::cpu_neon_available()) {
        try {
            dsmvc::CpuExecutor executor(dsmvc::CpuPath::neon);
            (void)executor;
        } catch (const std::runtime_error &) {
            goto neon_rejected;
        }
        throw std::runtime_error("unavailable explicit NEON path did not fail");
    }
neon_rejected:
    expect_invalid([] {
        dsmvc::CpuExecutor executor(static_cast<dsmvc::CpuPath>(999));
        (void)executor;
    }, "invalid CPU path enum");
}

} // namespace

int main() {
    try {
        test_strict_cpu_path_selection();
        test_cpu_paths_and_strides();
        test_accelerator_short_strides();
        test_checked_extreme_plan();
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "SIMD buffer contract test failed: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
