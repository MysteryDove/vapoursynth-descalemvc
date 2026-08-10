#include "metal_scheduler_apple.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using dsmvc::AxisPlan;
using dsmvc::AxisRequest;
using dsmvc::BorderMode;
using dsmvc::KernelKind;
using dsmvc::metal::Client;
using dsmvc::metal::FrameJob;
using dsmvc::metal::PlaneJob;
using dsmvc::metal::RunResult;

void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

[[nodiscard]] std::shared_ptr<const AxisPlan> make_plan(
    std::int32_t source, std::int32_t destination) {
    AxisRequest request;
    request.source_size = source;
    request.destination_size = destination;
    request.active_length = static_cast<double>(destination) - 0.25;
    request.shift = 0.125;
    request.kernel.kind = KernelKind::spline64;
    request.border = BorderMode::symmetric;
    auto plan = std::make_shared<const AxisPlan>(dsmvc::build_axis_plan(request));
    require(plan->valid() && plan->half_bandwidth >= 5,
            "wide Metal scheduler test plan is invalid");
    return plan;
}

[[nodiscard]] FrameJob horizontal_job(
    const std::shared_ptr<const AxisPlan> &plan,
    const float *source, float *destination, std::uint32_t height,
    std::shared_ptr<const void> source_lifetime = {}) {
    PlaneJob plane;
    plane.source = source;
    plane.source_stride_bytes =
        static_cast<std::ptrdiff_t>(plan->source_size * sizeof(float));
    plane.destination = destination;
    plane.destination_stride_bytes =
        static_cast<std::ptrdiff_t>(plan->destination_size * sizeof(float));
    plane.source_width = static_cast<std::uint32_t>(plan->source_size);
    plane.source_height = height;
    plane.destination_width = static_cast<std::uint32_t>(plan->destination_size);
    plane.destination_height = height;
    plane.sample_bytes = sizeof(float);
    plane.process_horizontal = true;
    plane.source_lifetime = std::move(source_lifetime);
    plane.horizontal = plan;

    FrameJob job;
    job.planes.push_back(std::move(plane));
    job.maximum_half_bandwidth =
        static_cast<std::uint32_t>(plan->half_bandwidth);
    return job;
}

[[nodiscard]] FrameJob two_axis_job(
    const std::shared_ptr<const AxisPlan> &horizontal,
    const std::shared_ptr<const AxisPlan> &vertical,
    const float *source, const std::vector<float *> &destinations,
    std::shared_ptr<const void> source_lifetime = {}) {
    FrameJob job;
    for (float *destination : destinations) {
        PlaneJob plane;
        plane.source = source;
        plane.source_stride_bytes =
            static_cast<std::ptrdiff_t>(horizontal->source_size * sizeof(float));
        plane.destination = destination;
        plane.destination_stride_bytes = static_cast<std::ptrdiff_t>(
            horizontal->destination_size * sizeof(float));
        plane.source_width =
            static_cast<std::uint32_t>(horizontal->source_size);
        plane.source_height = static_cast<std::uint32_t>(vertical->source_size);
        plane.destination_width =
            static_cast<std::uint32_t>(horizontal->destination_size);
        plane.destination_height =
            static_cast<std::uint32_t>(vertical->destination_size);
        plane.sample_bytes = sizeof(float);
        plane.process_horizontal = true;
        plane.process_vertical = true;
        plane.source_lifetime = source_lifetime;
        plane.horizontal = horizontal;
        plane.vertical = vertical;
        job.planes.push_back(std::move(plane));
    }
    job.maximum_half_bandwidth = static_cast<std::uint32_t>(std::max(
        horizontal->half_bandwidth, vertical->half_bandwidth));
    return job;
}

template <class Sample>
[[nodiscard]] FrameJob integer_two_axis_job(
    const std::shared_ptr<const AxisPlan> &horizontal,
    const std::shared_ptr<const AxisPlan> &vertical,
    const Sample *source, Sample *destination,
    const dsmvc::IntegerConversion &conversion,
    std::shared_ptr<const void> source_lifetime) {
    PlaneJob plane;
    plane.source = source;
    plane.source_stride_bytes = static_cast<std::ptrdiff_t>(
        horizontal->source_size * sizeof(Sample));
    plane.destination = destination;
    plane.destination_stride_bytes = static_cast<std::ptrdiff_t>(
        horizontal->destination_size * sizeof(Sample));
    plane.source_width = static_cast<std::uint32_t>(horizontal->source_size);
    plane.source_height = static_cast<std::uint32_t>(vertical->source_size);
    plane.destination_width = static_cast<std::uint32_t>(
        horizontal->destination_size);
    plane.destination_height = static_cast<std::uint32_t>(
        vertical->destination_size);
    plane.sample_bytes = sizeof(Sample);
    plane.integer_samples = true;
    plane.process_horizontal = true;
    plane.process_vertical = true;
    plane.conversion = conversion;
    plane.source_lifetime = std::move(source_lifetime);
    plane.horizontal = horizontal;
    plane.vertical = vertical;

    FrameJob job;
    job.planes.push_back(std::move(plane));
    job.maximum_half_bandwidth = static_cast<std::uint32_t>(std::max(
        horizontal->half_bandwidth, vertical->half_bandwidth));
    job.estimated_work = 1920ULL * 1080ULL * 2ULL;
    return job;
}

struct Outcome {
    RunResult result;
    std::exception_ptr error;
};

void solve_rows(
    const AxisPlan &plan, const std::vector<float> &source,
    std::vector<float> &destination, std::uint32_t height);

void solve_two_axis(
    const AxisPlan &horizontal, const AxisPlan &vertical,
    const std::vector<float> &source, std::vector<float> &destination);

[[nodiscard]] std::vector<Outcome> run_wave(
    const std::vector<std::shared_ptr<Client>> &clients,
    std::vector<FrameJob> jobs,
    std::vector<std::function<void()>> cpu_work, bool automatic = false) {
    require(clients.size() == jobs.size() && jobs.size() == cpu_work.size(),
            "invalid scheduler test wave");
    std::vector<Outcome> outcomes(jobs.size());
    std::barrier start(static_cast<std::ptrdiff_t>(jobs.size() + 1U));
    std::vector<std::thread> threads;
    threads.reserve(jobs.size());
    for (std::size_t index = 0; index < jobs.size(); ++index) {
        threads.emplace_back([&, index] {
            start.arrive_and_wait();
            try {
                outcomes[index].result = dsmvc::metal::run(
                    clients[index], std::move(jobs[index]),
                    std::move(cpu_work[index]), automatic);
            } catch (...) {
                outcomes[index].error = std::current_exception();
            }
        });
    }
    start.arrive_and_wait();
    for (auto &thread : threads) thread.join();
    return outcomes;
}

class RelaxedBatchTimeout final {
public:
    RelaxedBatchTimeout() {
        dsmvc::metal::set_batch_timeout_for_testing(20'000U);
    }
    ~RelaxedBatchTimeout() {
        dsmvc::metal::set_batch_timeout_for_testing(0U);
    }

    RelaxedBatchTimeout(const RelaxedBatchTimeout &) = delete;
    RelaxedBatchTimeout &operator=(const RelaxedBatchTimeout &) = delete;
};

[[nodiscard]] std::vector<Outcome> run_full_batch_wave(
    const std::vector<std::shared_ptr<Client>> &clients,
    std::vector<FrameJob> jobs,
    std::vector<std::function<void()>> cpu_work, bool automatic = false) {
    // Exact batch-shape assertions must not depend on hosted-runner core count.
    const RelaxedBatchTimeout timeout;
    return run_wave(
        clients, std::move(jobs), std::move(cpu_work), automatic);
}

constexpr std::uint64_t getfnative_spline36_work = 1280ULL * 720ULL * 7ULL;

void test_shared_input_automatic_admission(
    const std::shared_ptr<const AxisPlan> &plan,
    const std::vector<float> &source, std::uint32_t height) {
    constexpr std::size_t seed_count = 4U;
    constexpr std::size_t batch_count = 7U;
    const std::size_t output_size =
        static_cast<std::size_t>(plan->destination_size) * height;
    std::vector<std::vector<float>> outputs(
        batch_count, std::vector<float>(output_size));
    std::vector<std::vector<float>> expected(
        batch_count, std::vector<float>(output_size));
    std::vector<std::shared_ptr<Client>> clients;
    std::vector<FrameJob> jobs;
    std::vector<std::function<void()>> cpu_work;
    std::atomic<std::size_t> cpu_calls{0U};
    clients.reserve(seed_count + batch_count);
    for (std::size_t index = 0; index < seed_count + batch_count; ++index) {
        clients.push_back(dsmvc::metal::make_client());
    }

    std::vector<std::vector<float>> seed_outputs(
        seed_count, std::vector<float>(output_size));
    for (std::size_t index = 0; index < seed_count; ++index) {
        FrameJob job = horizontal_job(
            plan, source.data(), seed_outputs[index].data(), height);
        job.estimated_work = getfnative_spline36_work;
        bool used_cpu = false;
        const RunResult result = dsmvc::metal::run(
            clients[index], std::move(job), [&] {
                used_cpu = true;
                ++cpu_calls;
                solve_rows(*plan, source, seed_outputs[index], height);
            }, true);
        require(used_cpu && result.metal_batch_size == 0U,
                "serial shared-input census seed did not fall back to CPU");
    }

    jobs.reserve(batch_count);
    cpu_work.reserve(batch_count);
    for (std::size_t index = 0; index < batch_count; ++index) {
        FrameJob job = horizontal_job(
            plan, source.data(), outputs[index].data(), height);
        job.estimated_work = getfnative_spline36_work;
        jobs.push_back(std::move(job));
        cpu_work.emplace_back([&, index] {
            ++cpu_calls;
            solve_rows(*plan, source, outputs[index], height);
        });
        solve_rows(*plan, source, expected[index], height);
    }

    std::vector<std::shared_ptr<Client>> batch_clients(
        clients.begin() + static_cast<std::ptrdiff_t>(seed_count), clients.end());
    const auto outcomes = run_full_batch_wave(
        batch_clients, std::move(jobs), std::move(cpu_work), true);
    std::size_t metal_requests = 0U;
    for (std::size_t index = 0; index < batch_count; ++index) {
        require(!outcomes[index].error,
                "shared-input automatic request reported an error");
        if (outcomes[index].result.metal_batch_size != 0U) {
            ++metal_requests;
            require(outcomes[index].result.unique_input_planes == 1U,
                    "shared-input automatic batch lost source deduplication");
        }
        float maximum = 0.0F;
        for (std::size_t sample = 0; sample < output_size; ++sample) {
            maximum = std::max(
                maximum, std::abs(outputs[index][sample] - expected[index][sample]));
        }
        require(maximum <= 3.0e-6F,
                "shared-input automatic output differs from CPU");
    }
    require(metal_requests == batch_count,
            "recent cross-client fan-out did not form a full Metal batch");
    require(cpu_calls.load() == seed_count,
            "recent shared-input admission unexpectedly fell back after seeding");
}

void test_recent_input_expiry_and_key_isolation(
    const std::shared_ptr<const AxisPlan> &plan,
    const std::vector<float> &source, std::uint32_t height) {
    constexpr std::size_t seed_count = 4U;
    constexpr std::size_t probe_count = 2U;
    const std::size_t output_size =
        static_cast<std::size_t>(plan->destination_size) * height;
    std::vector<std::shared_ptr<Client>> clients;
    clients.reserve(seed_count);
    for (std::size_t index = 0; index < seed_count; ++index) {
        clients.push_back(dsmvc::metal::make_client());
    }

    std::vector<std::vector<float>> seed_outputs(
        seed_count, std::vector<float>(output_size));
    for (std::size_t index = 0; index < seed_count; ++index) {
        FrameJob job = horizontal_job(
            plan, source.data(), seed_outputs[index].data(), height);
        job.estimated_work = getfnative_spline36_work;
        const RunResult result = dsmvc::metal::run(
            clients[index], std::move(job),
            [&, index] { solve_rows(*plan, source, seed_outputs[index], height); },
            true);
        require(result.metal_batch_size == 0U,
                "recent-input isolation seed unexpectedly entered Metal");
    }

    auto run_probe = [&](const std::vector<float> &probe_source) {
        std::vector<std::vector<float>> outputs(
            probe_count, std::vector<float>(output_size));
        std::vector<std::shared_ptr<Client>> probe_clients(
            clients.begin(),
            clients.begin() + static_cast<std::ptrdiff_t>(probe_count));
        std::vector<FrameJob> jobs;
        std::vector<std::function<void()>> cpu_work;
        std::atomic<std::size_t> cpu_calls{0U};
        for (std::size_t index = 0; index < probe_count; ++index) {
            FrameJob job = horizontal_job(
                plan, probe_source.data(), outputs[index].data(), height);
            job.estimated_work = getfnative_spline36_work;
            jobs.push_back(std::move(job));
            cpu_work.emplace_back([&, index] {
                ++cpu_calls;
                solve_rows(*plan, probe_source, outputs[index], height);
            });
        }
        const auto outcomes = run_wave(
            probe_clients, std::move(jobs), std::move(cpu_work), true);
        for (const auto &outcome : outcomes) {
            require(!outcome.error && outcome.result.metal_batch_size == 0U,
                    "isolated recent-input probe unexpectedly entered Metal");
        }
        require(cpu_calls.load() == probe_count,
                "isolated recent-input probe skipped CPU fallback");
    };

    std::vector<float> alternate_source = source;
    alternate_source.front() += 0.125F;
    run_probe(alternate_source);

    std::this_thread::sleep_for(std::chrono::milliseconds{75});
    run_probe(source);
}

void test_automatic_admission_boundaries(
    const std::shared_ptr<Client> &client,
    const std::shared_ptr<const AxisPlan> &plan,
    const std::vector<float> &source, std::uint32_t height) {
    constexpr std::size_t count = 16U;
    const std::size_t output_size =
        static_cast<std::size_t>(plan->destination_size) * height;
    std::vector<std::vector<float>> outputs(
        count, std::vector<float>(output_size));
    std::vector<std::shared_ptr<Client>> clients(count, client);
    std::vector<FrameJob> jobs;
    std::vector<std::function<void()>> cpu_work;
    std::atomic<std::size_t> cpu_calls{0U};
    for (std::size_t index = 0; index < count; ++index) {
        FrameJob job = horizontal_job(
            plan, source.data(), outputs[index].data(), height);
        job.estimated_work = getfnative_spline36_work;
        jobs.push_back(std::move(job));
        cpu_work.emplace_back([&, index] {
            ++cpu_calls;
            solve_rows(*plan, source, outputs[index], height);
        });
    }
    const auto outcomes = run_wave(
        clients, std::move(jobs), std::move(cpu_work), true);
    for (const auto &outcome : outcomes) {
        require(!outcome.error && outcome.result.metal_batch_size == 0U,
                "one client incorrectly qualified for shared-input Metal");
    }
    require(cpu_calls.load() == count,
            "one-client automatic fallback did not execute every CPU job");

    std::vector<float> destination(output_size);
    bool used_cpu = false;
    FrameJob high_work = horizontal_job(
        plan, source.data(), destination.data(), height);
    high_work.estimated_work = 1920ULL * 1080ULL * 6ULL;
    const RunResult single = dsmvc::metal::run(
        client, std::move(high_work), [&] {
            used_cpu = true;
            solve_rows(*plan, source, destination, height);
        }, true);
    require(used_cpu && single.metal_batch_size == 0U,
            "a single automatic call bypassed the low-concurrency CPU gate");
}

void solve_rows(
    const AxisPlan &plan, const std::vector<float> &source,
    std::vector<float> &destination, std::uint32_t height) {
    for (std::uint32_t row = 0; row < height; ++row) {
        dsmvc::inverse_axis_f32(
            plan,
            source.data() + static_cast<std::size_t>(row) * plan.source_size,
            1,
            destination.data()
                + static_cast<std::size_t>(row) * plan.destination_size,
            1);
    }
}

void solve_two_axis(
    const AxisPlan &horizontal, const AxisPlan &vertical,
    const std::vector<float> &source, std::vector<float> &destination) {
    std::vector<float> intermediate(
        static_cast<std::size_t>(horizontal.destination_size)
        * vertical.source_size);
    solve_rows(
        horizontal, source, intermediate,
        static_cast<std::uint32_t>(vertical.source_size));
    for (std::int32_t column = 0; column < horizontal.destination_size;
         ++column) {
        dsmvc::inverse_axis_f32(
            vertical, intermediate.data() + column,
            horizontal.destination_size, destination.data() + column,
            horizontal.destination_size);
    }
}

void require_float_output(
    const std::vector<float> &actual, const std::vector<float> &expected,
    const std::string &label) {
    require(actual.size() == expected.size(), label + ": output size differs");
    float maximum = 0.0F;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        maximum = std::max(maximum, std::abs(actual[index] - expected[index]));
    }
    require(maximum <= 3.0e-6F, label + ": output differs from CPU");
}

void seed_shared_float_input(
    const std::vector<std::shared_ptr<Client>> &clients,
    const std::shared_ptr<const AxisPlan> &plan,
    const std::vector<float> &source, std::uint32_t height,
    const std::shared_ptr<const void> &source_lifetime) {
    require(clients.size() >= 4U, "resident seed requires four clients");
    std::vector<float> output(
        static_cast<std::size_t>(plan->destination_size) * height);
    for (std::size_t index = 0; index < 4U; ++index) {
        FrameJob job = horizontal_job(
            plan, source.data(), output.data(), height, source_lifetime);
        job.estimated_work = 1920ULL * 1080ULL * 2ULL;
        bool used_cpu = false;
        const RunResult result = dsmvc::metal::run(
            clients[index], std::move(job), [&] {
                used_cpu = true;
                solve_rows(*plan, source, output, height);
            }, true);
        require(used_cpu && result.metal_batch_size == 0U,
                "resident admission seed did not fall back to CPU");
    }
}

void test_resident_float_reuse_and_lifetime_identity(
    const std::shared_ptr<const AxisPlan> &plan, std::uint32_t height) {
    constexpr std::size_t count = 7U;
    auto source_owner = std::make_shared<std::vector<float>>(
        static_cast<std::size_t>(plan->source_size) * height);
    for (std::size_t index = 0; index < source_owner->size(); ++index) {
        (*source_owner)[index] = static_cast<float>((index * 29U + 3U) & 2047U)
            / 2047.0F;
    }
    std::weak_ptr<std::vector<float>> retained_source = source_owner;
    auto token_a = std::make_shared<
        std::shared_ptr<std::vector<float>>>(source_owner);
    std::shared_ptr<const void> lifetime_a = token_a;

    std::vector<std::shared_ptr<Client>> clients;
    clients.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        clients.push_back(dsmvc::metal::make_client());
    }
    seed_shared_float_input(clients, plan, *source_owner, height, lifetime_a);

    const std::size_t output_size =
        static_cast<std::size_t>(plan->destination_size) * height;
    std::vector<float> expected(output_size);
    solve_rows(*plan, *source_owner, expected, height);
    auto run_cached_wave = [&](const std::shared_ptr<const void> &lifetime) {
        std::vector<std::vector<float>> outputs(
            count, std::vector<float>(output_size));
        std::vector<FrameJob> jobs;
        std::vector<std::function<void()>> cpu_work;
        for (std::size_t index = 0; index < count; ++index) {
            FrameJob job = horizontal_job(
                plan, source_owner->data(), outputs[index].data(), height,
                lifetime);
            job.estimated_work = 1920ULL * 1080ULL * 2ULL;
            jobs.push_back(std::move(job));
            cpu_work.emplace_back([&, index] { outputs[index] = expected; });
        }
        auto outcomes = run_full_batch_wave(
            clients, std::move(jobs), std::move(cpu_work), true);
        for (std::size_t index = 0; index < count; ++index) {
            require(!outcomes[index].error
                        && outcomes[index].result.metal_batch_size == count,
                    "resident Float32 wave did not enter Metal");
            require_float_output(
                outputs[index], expected, "resident Float32");
        }
        return outcomes;
    };

    const auto first = run_cached_wave(lifetime_a);
    const std::size_t source_bytes = source_owner->size() * sizeof(float);
    require(first.front().result.resident_producers == 1U
                && first.front().result.resident_hits == count - 1U,
            "first resident wave producer/hit accounting is incorrect");
    require(first.front().result.eliminated_staging_bytes
                == (count - 1U) * source_bytes,
            "first resident wave eliminated-byte accounting is incorrect");

    const auto second = run_cached_wave(lifetime_a);
    require(second.front().result.resident_producers == 0U
                && second.front().result.resident_hits == count,
            "cross-submission resident reuse did not hit every consumer");
    require(second.front().result.eliminated_staging_bytes
                == count * source_bytes,
            "cross-submission eliminated-byte accounting is incorrect");

    for (float &sample : *source_owner) sample = 1.0F - sample;
    solve_rows(*plan, *source_owner, expected, height);
    auto token_b = std::make_shared<
        std::shared_ptr<std::vector<float>>>(source_owner);
    std::shared_ptr<const void> lifetime_b = token_b;
    seed_shared_float_input(clients, plan, *source_owner, height, lifetime_b);
    const auto reused_address = run_cached_wave(lifetime_b);
    require(reused_address.front().result.resident_producers == 1U,
            "same source address with a new lifetime reused stale resident data");

    source_owner.reset();
    token_a.reset();
    token_b.reset();
    lifetime_a.reset();
    lifetime_b.reset();
    require(!retained_source.expired(),
            "resident cache did not retain its source lifetime");

    const auto before_close = dsmvc::metal::diagnostics();
    for (std::size_t index = 0; index + 1U < clients.size(); ++index) {
        clients[index]->close();
    }
    const auto after_partial_close = dsmvc::metal::diagnostics();
    require(!retained_source.expired(),
            "closing one of multiple resident owners released the source");
    require(after_partial_close.resident_cache_entries
                    == before_close.resident_cache_entries
                && after_partial_close.resident_cache_bytes
                    == before_close.resident_cache_bytes,
            "partial resident-owner close evicted a shared entry");

    clients.back()->close();
    const auto after_final_close = dsmvc::metal::diagnostics();
    require(retained_source.expired(),
            "closing the final resident owner retained the source");
    require(after_final_close.resident_cache_entries + 2U
                    == before_close.resident_cache_entries
                && after_final_close.resident_cache_bytes
                    < before_close.resident_cache_bytes,
            "final resident-owner close did not purge both lifetime keys");

    // Close is part of the host filter destructor path and must be repeatable.
    clients.back()->close();
}

void test_resident_ready_singleton_after_recent_input_expiry(
    const std::shared_ptr<const AxisPlan> &plan, std::uint32_t height) {
    constexpr std::size_t count = 7U;
    auto source = std::make_shared<std::vector<float>>(
        static_cast<std::size_t>(plan->source_size) * height);
    for (std::size_t index = 0; index < source->size(); ++index) {
        (*source)[index] = static_cast<float>((index * 43U + 17U) & 2047U)
            / 2047.0F;
    }
    auto token = std::make_shared<std::shared_ptr<std::vector<float>>>(source);
    std::shared_ptr<const void> lifetime = token;
    std::vector<std::shared_ptr<Client>> clients;
    clients.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        clients.push_back(dsmvc::metal::make_client());
    }
    seed_shared_float_input(clients, plan, *source, height, lifetime);

    const std::size_t output_size =
        static_cast<std::size_t>(plan->destination_size) * height;
    std::vector<float> expected(output_size);
    solve_rows(*plan, *source, expected, height);
    std::vector<std::vector<float>> outputs(
        count, std::vector<float>(output_size));
    std::vector<FrameJob> jobs;
    std::vector<std::function<void()>> cpu_work;
    for (std::size_t index = 0; index < count; ++index) {
        FrameJob job = horizontal_job(
            plan, source->data(), outputs[index].data(), height, lifetime);
        job.estimated_work = getfnative_spline36_work;
        jobs.push_back(std::move(job));
        cpu_work.emplace_back([&, index] { outputs[index] = expected; });
    }
    const auto producer_wave = run_full_batch_wave(
        clients, std::move(jobs), std::move(cpu_work), true);
    for (std::size_t index = 0; index < count; ++index) {
        require(!producer_wave[index].error
                    && producer_wave[index].result.metal_batch_size == count,
                "resident singleton seed wave did not enter Metal");
        require_float_output(
            outputs[index], expected, "resident singleton seed wave");
    }

    std::this_thread::sleep_for(std::chrono::milliseconds{75});
    std::vector<float> singleton_output(output_size);
    bool singleton_used_cpu = false;
    FrameJob singleton = horizontal_job(
        plan, source->data(), singleton_output.data(), height, lifetime);
    singleton.estimated_work = getfnative_spline36_work;
    const RunResult singleton_result = dsmvc::metal::run(
        clients.front(), std::move(singleton), [&] {
            singleton_used_cpu = true;
            singleton_output = expected;
        }, true);
    require(!singleton_used_cpu && singleton_result.metal_batch_size == 1U,
            "ready resident singleton did not use available Metal capacity");
    require(singleton_result.resident_producers == 0U
                && singleton_result.resident_hits == 1U,
            "ready resident singleton did not reuse its cached input");
    require_float_output(singleton_output, expected, "resident singleton");

    auto alternate = std::make_shared<std::vector<float>>(*source);
    alternate->front() += 0.25F;
    auto alternate_token = std::make_shared<
        std::shared_ptr<std::vector<float>>>(alternate);
    std::shared_ptr<const void> alternate_lifetime = alternate_token;
    std::vector<float> alternate_expected(output_size);
    solve_rows(*plan, *alternate, alternate_expected, height);
    std::vector<float> alternate_output(output_size);
    bool alternate_used_cpu = false;
    FrameJob nonresident = horizontal_job(
        plan, alternate->data(), alternate_output.data(), height,
        alternate_lifetime);
    nonresident.estimated_work = getfnative_spline36_work;
    const RunResult nonresident_result = dsmvc::metal::run(
        clients.front(), std::move(nonresident), [&] {
            alternate_used_cpu = true;
            alternate_output = alternate_expected;
        }, true);
    require(alternate_used_cpu && nonresident_result.metal_batch_size == 0U,
            "nonresident automatic singleton bypassed the CPU fallback");
    require_float_output(
        alternate_output, alternate_expected, "nonresident singleton");

    for (const auto &client : clients) client->close();
}

void test_resident_producer_failure_recovery(
    const std::shared_ptr<const AxisPlan> &plan, std::uint32_t height) {
    constexpr std::size_t count = 7U;
    auto source = std::make_shared<std::vector<float>>(
        static_cast<std::size_t>(plan->source_size) * height, 0.375F);
    auto token = std::make_shared<std::shared_ptr<std::vector<float>>>(source);
    std::shared_ptr<const void> lifetime = token;
    std::vector<std::shared_ptr<Client>> clients;
    for (std::size_t index = 0; index < count; ++index) {
        clients.push_back(dsmvc::metal::make_client());
    }
    seed_shared_float_input(clients, plan, *source, height, lifetime);

    const std::size_t output_size =
        static_cast<std::size_t>(plan->destination_size) * height;
    std::vector<float> expected(output_size);
    solve_rows(*plan, *source, expected, height);
    auto make_wave = [&](std::vector<std::vector<float>> &outputs) {
        std::vector<FrameJob> jobs;
        std::vector<std::function<void()>> cpu_work;
        for (std::size_t index = 0; index < count; ++index) {
            FrameJob job = horizontal_job(
                plan, source->data(), outputs[index].data(), height, lifetime);
            job.estimated_work = 1920ULL * 1080ULL * 2ULL;
            jobs.push_back(std::move(job));
            cpu_work.emplace_back([&, index] { outputs[index] = expected; });
        }
        return run_full_batch_wave(
            clients, std::move(jobs), std::move(cpu_work), true);
    };

    const auto before = dsmvc::metal::diagnostics();
    dsmvc::metal::fail_next_resident_producer_for_testing();
    std::vector<std::vector<float>> failed_outputs(
        count, std::vector<float>(output_size));
    const auto failed = make_wave(failed_outputs);
    for (std::size_t index = 0; index < count; ++index) {
        require(!failed[index].error
                    && failed[index].result.metal_batch_size == 0U,
                "automatic resident producer failure did not fall back to CPU");
        require_float_output(
            failed_outputs[index], expected, "resident producer fallback");
    }
    const auto after_failure = dsmvc::metal::diagnostics();
    require(after_failure.metal_errors >= before.metal_errors + 1U
                && after_failure.consecutive_metal_errors >= 1U,
            "resident producer failure was not observable");

    std::vector<std::vector<float>> retry_outputs(
        count, std::vector<float>(output_size));
    const auto retry = make_wave(retry_outputs);
    for (std::size_t index = 0; index < count; ++index) {
        require(!retry[index].error
                    && retry[index].result.resident_producers == 1U,
                "resident producer retry did not rebuild the cache entry");
        require_float_output(
            retry_outputs[index], expected, "resident producer retry");
    }
    const auto after_retry = dsmvc::metal::diagnostics();
    require(after_retry.consecutive_metal_errors == 0U
                && after_retry.maximum_consecutive_metal_errors >= 1U,
            "successful resident retry did not clear the error streak");
}

template <class Sample>
void test_resident_integer_conversion_keys(
    const std::shared_ptr<const AxisPlan> &horizontal,
    const std::shared_ptr<const AxisPlan> &vertical,
    const dsmvc::IntegerConversion &first_conversion,
    const dsmvc::IntegerConversion &second_conversion) {
    constexpr std::size_t count = 7U;
    const std::size_t source_size = static_cast<std::size_t>(
        horizontal->source_size) * vertical->source_size;
    auto source = std::make_shared<std::vector<Sample>>(source_size);
    const std::uint32_t modulus = std::max(
        first_conversion.output_maximum,
        second_conversion.output_maximum) + 1U;
    for (std::size_t index = 0; index < source->size(); ++index) {
        (*source)[index] = static_cast<Sample>((index * 41U + 13U) % modulus);
    }
    auto token = std::make_shared<std::shared_ptr<std::vector<Sample>>>(source);
    std::shared_ptr<const void> lifetime = token;

    const std::size_t output_size = static_cast<std::size_t>(
        horizontal->destination_size) * vertical->destination_size;
    std::array<std::vector<Sample>, 2> expected{
        std::vector<Sample>(output_size), std::vector<Sample>(output_size)};
    dsmvc::CpuExecutor reference(dsmvc::CpuPath::scalar);
    reference.prepare(horizontal);
    reference.prepare(vertical);
    reference.seal();
    if constexpr (std::is_same_v<Sample, std::uint8_t>) {
        reference.inverse_2d_u8(
            *horizontal, *vertical, source->data(), horizontal->source_size,
            expected[0].data(), horizontal->destination_size, first_conversion);
        reference.inverse_2d_u8(
            *horizontal, *vertical, source->data(), horizontal->source_size,
            expected[1].data(), horizontal->destination_size, second_conversion);
    } else {
        reference.inverse_2d_u16(
            *horizontal, *vertical, source->data(), horizontal->source_size,
            expected[0].data(), horizontal->destination_size, first_conversion);
        reference.inverse_2d_u16(
            *horizontal, *vertical, source->data(), horizontal->source_size,
            expected[1].data(), horizontal->destination_size, second_conversion);
    }

    std::vector<std::shared_ptr<Client>> clients;
    for (std::size_t index = 0; index < count; ++index) {
        clients.push_back(dsmvc::metal::make_client());
    }
    for (std::size_t client_index = 0; client_index < 4U; ++client_index) {
        for (std::size_t conversion_index = 0; conversion_index < 2U;
             ++conversion_index) {
            std::vector<Sample> seed_output(output_size);
            const auto &conversion = conversion_index == 0U
                ? first_conversion : second_conversion;
            bool used_cpu = false;
            const RunResult result = dsmvc::metal::run(
                clients[client_index], integer_two_axis_job(
                    horizontal, vertical, source->data(), seed_output.data(),
                    conversion, lifetime), [&] {
                    used_cpu = true;
                    seed_output = expected[conversion_index];
                }, true);
            require(used_cpu && result.metal_batch_size == 0U,
                    "integer resident admission seed did not use CPU");
        }
    }

    std::vector<std::vector<Sample>> outputs(
        count, std::vector<Sample>(output_size));
    std::vector<FrameJob> jobs;
    std::vector<std::function<void()>> cpu_work;
    for (std::size_t index = 0; index < count; ++index) {
        const std::size_t conversion_index = index % 2U;
        const auto &conversion = conversion_index == 0U
            ? first_conversion : second_conversion;
        jobs.push_back(integer_two_axis_job(
            horizontal, vertical, source->data(), outputs[index].data(),
            conversion, lifetime));
        cpu_work.emplace_back([&, index, conversion_index] {
            outputs[index] = expected[conversion_index];
        });
    }
    const auto outcomes = run_full_batch_wave(
        clients, std::move(jobs), std::move(cpu_work), true);
    for (std::size_t index = 0; index < count; ++index) {
        require(!outcomes[index].error
                    && outcomes[index].result.metal_batch_size == count,
                "integer resident conversion wave did not enter Metal");
        require(outcomes[index].result.resident_producers == 2U
                    && outcomes[index].result.resident_hits == count - 2U,
                "integer conversion keys aliased resident cache entries");
        const auto &wanted = expected[index % 2U];
        for (std::size_t sample = 0; sample < output_size; ++sample) {
            const std::uint32_t actual_value = outputs[index][sample];
            const std::uint32_t wanted_value = wanted[sample];
            const std::uint32_t difference = actual_value > wanted_value
                ? actual_value - wanted_value : wanted_value - actual_value;
            require(difference <= 1U,
                    "integer resident output differs by more than one LSB");
        }
    }
}

void test_resident_in_flight_eviction_safety() {
    constexpr std::size_t count = 16U;
    constexpr std::size_t wave_size = count / 2U;
    constexpr std::uint32_t height = 1024U;
    const auto plan = make_plan(1024, 960);
    const std::size_t source_size =
        static_cast<std::size_t>(plan->source_size) * height;
    const std::size_t output_size =
        static_cast<std::size_t>(plan->destination_size) * height;
    std::vector<std::shared_ptr<std::vector<float>>> sources;
    std::vector<std::shared_ptr<const void>> lifetimes;
    std::vector<std::vector<float>> outputs(
        count, std::vector<float>(output_size));
    std::vector<std::shared_ptr<Client>> clients;
    for (std::size_t index = 0; index < 7U; ++index) {
        clients.push_back(dsmvc::metal::make_client());
    }
    for (std::size_t request = 0; request < count; ++request) {
        auto source = std::make_shared<std::vector<float>>(source_size);
        for (std::size_t index = 0; index < source_size; ++index) {
            (*source)[index] = static_cast<float>(
                (index * 17U + request * 31U + 5U) & 4095U) / 4095.0F;
        }
        auto token = std::make_shared<
            std::shared_ptr<std::vector<float>>>(source);
        std::shared_ptr<const void> lifetime = token;
        sources.push_back(source);
        lifetimes.push_back(lifetime);
    }

    const auto before = dsmvc::metal::diagnostics();
    std::vector<Outcome> outcomes;
    outcomes.reserve(count);
    for (std::size_t wave = 0; wave < count; wave += wave_size) {
        std::vector<std::shared_ptr<Client>> wave_clients;
        std::vector<FrameJob> jobs;
        std::vector<std::function<void()>> cpu_work;
        for (std::size_t request = wave; request < wave + wave_size; ++request) {
            std::vector<float> seed_output(output_size);
            for (std::size_t client_index = 0; client_index < 4U; ++client_index) {
                FrameJob seed = horizontal_job(
                    plan, sources[request]->data(), seed_output.data(), height,
                    lifetimes[request]);
                seed.estimated_work = 1920ULL * 1080ULL * 2ULL;
                const RunResult result = dsmvc::metal::run(
                    clients[client_index], std::move(seed), [] {}, true);
                require(result.metal_batch_size == 0U,
                        "resident pressure admission seed entered Metal");
            }
            wave_clients.push_back(clients[(request - wave) % clients.size()]);
            FrameJob job = horizontal_job(
                plan, sources[request]->data(), outputs[request].data(), height,
                lifetimes[request]);
            job.estimated_work = 1920ULL * 1080ULL * 2ULL;
            jobs.push_back(std::move(job));
            cpu_work.emplace_back([&, request] {
                solve_rows(*plan, *sources[request], outputs[request], height);
            });
        }
        auto wave_outcomes = run_wave(
            wave_clients, std::move(jobs), std::move(cpu_work), true);
        outcomes.insert(
            outcomes.end(),
            std::make_move_iterator(wave_outcomes.begin()),
            std::make_move_iterator(wave_outcomes.end()));
    }
    std::size_t metal_requests = 0U;
    for (std::size_t request = 0; request < count; ++request) {
        require(!outcomes[request].error,
                "resident pressure request reported an error");
        metal_requests += outcomes[request].result.metal_batch_size != 0U;
        std::vector<float> expected(output_size);
        solve_rows(*plan, *sources[request], expected, height);
        require_float_output(
            outputs[request], expected, "resident pressure output");
    }
    const auto after = dsmvc::metal::diagnostics();
    require(metal_requests >= 7U,
            "resident pressure did not produce a complete Metal batch");
    require(after.resident_cache_capacity == 16U * 1024U * 1024U
                && after.resident_cache_bytes <= after.resident_cache_capacity,
            "resident cache exceeded its configured bounded capacity");
    require(after.resident_cache_pinned_eviction_blocks
                > before.resident_cache_pinned_eviction_blocks,
            "resident pressure did not protect submission-pinned entries");
    require(after.resident_cache_evictions > before.resident_cache_evictions,
            "resident pressure did not exercise LRU eviction");
}

void align_wide_gpu_batch_window(
    const std::shared_ptr<Client> &client,
    const std::shared_ptr<const AxisPlan> &plan,
    const std::vector<float> &source, std::uint32_t height) {
    // Drain the 9-decision GPU window so the next 7 calls form one full batch.
    constexpr std::size_t decisions_before_full_batch = 9U;
    std::vector<float> destination(
        static_cast<std::size_t>(plan->destination_size) * height);
    for (std::size_t index = 0; index < decisions_before_full_batch; ++index) {
        const RunResult result = dsmvc::metal::run(
            client, horizontal_job(plan, source.data(), destination.data(), height),
            [&] { solve_rows(*plan, source, destination, height); }, false);
        require(result.metal_batch_size == 0U,
                "wide scheduler singleton unexpectedly entered Metal");
    }
}

void test_shared_source_and_recovery(
    const std::shared_ptr<Client> &first,
    const std::shared_ptr<Client> &second,
    const std::shared_ptr<const AxisPlan> &plan,
    const std::vector<float> &source, std::uint32_t height) {
    align_wide_gpu_batch_window(first, plan, source, height);

    constexpr std::size_t count = 7U;
    const std::size_t output_size =
        static_cast<std::size_t>(plan->destination_size) * height;
    std::vector<std::vector<float>> outputs(
        count, std::vector<float>(output_size));
    std::vector<std::vector<float>> expected(
        count, std::vector<float>(output_size));
    std::vector<std::shared_ptr<Client>> clients;
    std::vector<FrameJob> jobs;
    std::vector<std::function<void()>> cpu_work;
    for (std::size_t index = 0; index < count; ++index) {
        clients.push_back(index % 2U == 0U ? first : second);
        jobs.push_back(horizontal_job(
            plan, source.data(), outputs[index].data(), height));
        cpu_work.emplace_back([&, index] {
            solve_rows(*plan, source, outputs[index], height);
        });
        solve_rows(*plan, source, expected[index], height);
    }

    const auto outcomes = run_full_batch_wave(
        clients, std::move(jobs), std::move(cpu_work));
    for (std::size_t index = 0; index < count; ++index) {
        require(!outcomes[index].error, "valid Metal batch reported an error");
        require(outcomes[index].result.metal_batch_size == count,
                "wide GPU batch did not collect seven requests");
        require(outcomes[index].result.unique_input_planes == 1U,
                "cross-client source plane was uploaded more than once");
        require(outcomes[index].result.staging_memcpy_calls == count + 1U,
                "cross-client staging copy count is not deduplicated");
        require(outcomes[index].result.axis_dispatches == 1U
                    && outcomes[index].result.conversion_dispatches == 0U,
                "compatible shared-source jobs were not encoded as one batch");
        float maximum = 0.0F;
        for (std::size_t sample = 0; sample < output_size; ++sample) {
            maximum = std::max(
                maximum, std::abs(outputs[index][sample] - expected[index][sample]));
        }
        require(maximum <= 3.0e-6F,
                "direct scheduler Float32 output differs from CPU");
    }
}

void test_ring_backpressure(
    const std::shared_ptr<Client> &client,
    const std::shared_ptr<const AxisPlan> &horizontal,
    const std::shared_ptr<const AxisPlan> &vertical,
    const std::vector<float> &source) {
    constexpr std::size_t request_count = 64U;
    constexpr std::size_t plane_count = 4U;
    const std::size_t plane_samples =
        static_cast<std::size_t>(horizontal->destination_size)
        * vertical->destination_size;
    std::vector<std::vector<std::vector<float>>> outputs(
        request_count,
        std::vector<std::vector<float>>(
            plane_count, std::vector<float>(plane_samples)));
    std::vector<std::shared_ptr<Client>> clients(request_count, client);
    std::vector<FrameJob> jobs;
    std::vector<std::function<void()>> cpu_work;
    jobs.reserve(request_count);
    cpu_work.reserve(request_count);
    for (std::size_t request = 0; request < request_count; ++request) {
        std::vector<float *> destinations;
        for (auto &plane : outputs[request]) destinations.push_back(plane.data());
        jobs.push_back(two_axis_job(
            horizontal, vertical, source.data(), destinations));
        cpu_work.emplace_back([] {});
    }

    const auto before = dsmvc::metal::diagnostics();
    const auto outcomes = run_wave(clients, std::move(jobs), std::move(cpu_work));
    std::size_t metal_requests = 0;
    for (const auto &outcome : outcomes) {
        require(!outcome.error, "ring stress request failed");
        metal_requests += outcome.result.metal_batch_size != 0U;
    }
    const auto after = dsmvc::metal::diagnostics();
    require(metal_requests >= 21U,
            "ring stress did not submit enough Metal requests");
    require(after.submissions >= before.submissions + 3U,
            "ring stress did not create multiple command buffers");
    require(after.submissions == after.completions,
            "command buffer slot remained occupied after completion");
    require(after.ring_slots == 3U && after.maximum_in_flight >= 2U
                && after.maximum_in_flight <= after.ring_slots,
            "three-slot asynchronous ring/backpressure was not exercised");
    require(after.plan_cache_entries <= 128U && after.plan_cache_bytes != 0U,
            "Metal plan cache is unbounded or empty after use");
}

void test_interleaved_plan_offsets(
    const std::shared_ptr<Client> &client,
    const std::shared_ptr<const AxisPlan> &primary,
    const std::vector<float> &first_source,
    const std::vector<float> &second_source, std::uint32_t height) {
    align_wide_gpu_batch_window(client, primary, first_source, height);

    constexpr std::size_t count = 7U;
    std::vector<std::shared_ptr<const AxisPlan>> plans;
    plans.reserve(count);
    plans.push_back(make_plan(primary->source_size, primary->destination_size));
    plans.push_back(primary);
    plans.push_back(primary);
    for (std::size_t index = plans.size(); index < count; ++index) {
        plans.push_back(make_plan(primary->source_size, primary->destination_size));
    }

    std::vector<std::vector<float>> outputs(
        count, std::vector<float>(
                   static_cast<std::size_t>(primary->destination_size) * height));
    std::vector<std::vector<float>> expected(
        count, std::vector<float>(
                   static_cast<std::size_t>(primary->destination_size) * height));
    std::vector<std::shared_ptr<Client>> clients(count, client);
    std::vector<FrameJob> jobs;
    std::vector<std::function<void()>> cpu_work;
    for (std::size_t index = 0; index < count; ++index) {
        const std::vector<float> &source = index == 1U
            ? second_source : first_source;
        jobs.push_back(horizontal_job(
            plans[index], source.data(), outputs[index].data(), height));
        cpu_work.emplace_back([] {});
        solve_rows(*plans[index], source, expected[index], height);
    }

    const auto outcomes = run_full_batch_wave(
        clients, std::move(jobs), std::move(cpu_work));
    for (std::size_t index = 0; index < count; ++index) {
        require(!outcomes[index].error,
                "interleaved plan/input offsets rejected a valid submission");
        require(outcomes[index].result.metal_batch_size == count,
                "interleaved scheduler wave did not enter Metal");
        require(outcomes[index].result.axis_dispatches == 1U,
                "interleaved plans were not merged into one dispatch");
        require(outcomes[index].result.heterogeneous_axis_dispatches == 1U
                    && outcomes[index].result.heterogeneous_axis_descriptors >= 2U,
                "interleaved plans did not report heterogeneous descriptors");
        float maximum = 0.0F;
        for (std::size_t sample = 0; sample < outputs[index].size(); ++sample) {
            maximum = std::max(maximum, std::abs(
                outputs[index][sample] - expected[index][sample]));
        }
        require(maximum <= 3.0e-6F,
                "interleaved plan output differs from CPU");
    }
}

void test_heterogeneous_plan_geometry(
    const std::shared_ptr<Client> &client,
    const std::shared_ptr<const AxisPlan> &primary,
    const std::vector<float> &source, std::uint32_t height) {
    align_wide_gpu_batch_window(client, primary, source, height);

    constexpr std::size_t count = 7U;
    std::vector<std::shared_ptr<const AxisPlan>> plans;
    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<float>> expected;
    std::vector<FrameJob> jobs;
    std::vector<std::function<void()>> cpu_work;
    plans.reserve(count);
    outputs.reserve(count);
    expected.reserve(count);
    jobs.reserve(count);
    cpu_work.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        plans.push_back(make_plan(
            primary->source_size,
            primary->destination_size - static_cast<std::int32_t>(index)));
        const std::size_t output_size =
            static_cast<std::size_t>(plans.back()->destination_size) * height;
        outputs.emplace_back(output_size);
        expected.emplace_back(output_size);
        jobs.push_back(horizontal_job(
            plans.back(), source.data(), outputs.back().data(), height));
        cpu_work.emplace_back([] {});
        solve_rows(*plans.back(), source, expected.back(), height);
    }

    std::vector<std::shared_ptr<Client>> clients(count, client);
    const auto outcomes = run_full_batch_wave(
        clients, std::move(jobs), std::move(cpu_work));
    for (std::size_t index = 0; index < count; ++index) {
        require(!outcomes[index].error,
                "heterogeneous geometry request failed");
        require(outcomes[index].result.metal_batch_size == count,
                "heterogeneous geometry wave did not enter Metal");
        require(outcomes[index].result.unique_input_planes == 1U,
                "heterogeneous geometry wave duplicated its source upload");
        require(outcomes[index].result.axis_dispatches == 1U,
                "heterogeneous geometry plans were not merged into one dispatch");
        require(outcomes[index].result.heterogeneous_axis_dispatches == 1U
                    && outcomes[index].result.heterogeneous_axis_descriptors
                        == count,
                "heterogeneous geometry descriptors were not observable");
        float maximum = 0.0F;
        for (std::size_t sample = 0; sample < outputs[index].size(); ++sample) {
            maximum = std::max(maximum, std::abs(
                outputs[index][sample] - expected[index][sample]));
        }
        require(maximum <= 3.0e-6F,
                "heterogeneous geometry output differs from CPU");
    }
}

void test_heterogeneous_two_axis_geometry(
    const std::shared_ptr<Client> &client,
    const std::shared_ptr<const AxisPlan> &primary_horizontal,
    const std::vector<float> &source, std::uint32_t source_height) {
    align_wide_gpu_batch_window(
        client, primary_horizontal, source, source_height);

    constexpr std::size_t count = 7U;
    const auto primary_vertical = make_plan(
        static_cast<std::int32_t>(source_height),
        static_cast<std::int32_t>(source_height) - 4);
    std::vector<std::shared_ptr<const AxisPlan>> horizontal_plans;
    std::vector<std::shared_ptr<const AxisPlan>> vertical_plans;
    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<float>> expected;
    std::vector<FrameJob> jobs;
    std::vector<std::function<void()>> cpu_work;
    horizontal_plans.reserve(count);
    vertical_plans.reserve(count);
    outputs.reserve(count);
    expected.reserve(count);
    jobs.reserve(count);
    cpu_work.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        horizontal_plans.push_back(make_plan(
            primary_horizontal->source_size,
            primary_horizontal->destination_size
                - static_cast<std::int32_t>(index)));
        vertical_plans.push_back(make_plan(
            primary_vertical->source_size,
            primary_vertical->destination_size
                - static_cast<std::int32_t>(index)));
        const std::size_t output_size = static_cast<std::size_t>(
            horizontal_plans.back()->destination_size)
            * vertical_plans.back()->destination_size;
        outputs.emplace_back(output_size);
        expected.emplace_back(output_size);
        jobs.push_back(two_axis_job(
            horizontal_plans.back(), vertical_plans.back(), source.data(),
            std::vector<float *>{outputs.back().data()}));
        cpu_work.emplace_back([] {});
        solve_two_axis(
            *horizontal_plans.back(), *vertical_plans.back(),
            source, expected.back());
    }

    std::vector<std::shared_ptr<Client>> clients(count, client);
    const auto outcomes = run_full_batch_wave(
        clients, std::move(jobs), std::move(cpu_work));
    for (std::size_t index = 0; index < count; ++index) {
        require(!outcomes[index].error,
                "heterogeneous two-axis request failed");
        require(outcomes[index].result.metal_batch_size == count,
                "heterogeneous two-axis wave did not enter Metal");
        require(outcomes[index].result.unique_input_planes == 1U,
                "heterogeneous two-axis wave duplicated its source upload");
        require(outcomes[index].result.axis_dispatches == 2U,
                "heterogeneous two-axis plans were not merged into two dispatches");
        require(outcomes[index].result.heterogeneous_axis_dispatches == 2U
                    && outcomes[index].result.heterogeneous_axis_descriptors
                        == count * 2U,
                "heterogeneous two-axis descriptors were not observable");
        float maximum = 0.0F;
        for (std::size_t sample = 0; sample < outputs[index].size(); ++sample) {
            maximum = std::max(maximum, std::abs(
                outputs[index][sample] - expected[index][sample]));
        }
        require(maximum <= 3.0e-6F,
                "heterogeneous two-axis output differs from CPU");
    }
}

void test_error_propagation(
    const std::shared_ptr<Client> &client,
    const std::shared_ptr<const AxisPlan> &plan,
    const std::vector<float> &source, std::uint32_t height) {
    align_wide_gpu_batch_window(client, plan, source, height);

    constexpr std::size_t count = 7U;
    std::vector<std::shared_ptr<Client>> clients(count, client);
    std::vector<FrameJob> jobs;
    std::vector<std::function<void()>> cpu_work;
    std::vector<float> destinations(
        count * static_cast<std::size_t>(plan->destination_size) * height);
    for (std::size_t index = 0; index < count; ++index) {
        FrameJob job = horizontal_job(
            plan, nullptr,
            destinations.data()
                + index * static_cast<std::size_t>(plan->destination_size) * height,
            height);
        jobs.push_back(std::move(job));
        cpu_work.emplace_back([] {});
    }
    const auto outcomes = run_full_batch_wave(
        clients, std::move(jobs), std::move(cpu_work));
    const auto failures = std::count_if(
        outcomes.begin(), outcomes.end(), [](const Outcome &outcome) {
            return outcome.error != nullptr;
        });
    require(failures == 7U,
            "invalid GPU submission was not propagated to its full batch");
}

void test_float64_plan_cpu_fallback(const std::shared_ptr<Client> &client) {
    AxisRequest request;
    request.source_size = 1080;
    request.destination_size = 980;
    request.active_length = 978.1;
    request.shift = 0.95;
    request.kernel.kind = KernelKind::lanczos;
    request.kernel.taps = 2;
    auto plan = std::make_shared<const AxisPlan>(
        dsmvc::build_axis_plan(request));
    require(plan->requires_float64(),
            "Metal Float64 fallback fixture did not select Float64");

    std::vector<float> source(static_cast<std::size_t>(plan->source_size));
    std::vector<float> destination(
        static_cast<std::size_t>(plan->destination_size));
    for (std::size_t index = 0; index < source.size(); ++index) {
        source[index] = static_cast<float>((index * 37U + 11U) & 4095U)
            / 4095.0F;
    }
    bool used_cpu = false;
    const RunResult result = dsmvc::metal::run(
        client,
        horizontal_job(plan, source.data(), destination.data(), 1U),
        [&] {
            used_cpu = true;
            solve_rows(*plan, source, destination, 1U);
        },
        false);
    require(used_cpu && result.metal_batch_size == 0U,
            "Float64 axis plan entered the Float32 Metal path");
    require(std::all_of(
                destination.begin(), destination.end(),
                [](float value) { return std::isfinite(value); }),
            "Float64 Metal CPU fallback produced non-finite output");
}

} // namespace

int main() {
    try {
        require(::setenv("DSMVC_METAL_RESIDENT_MB", "16", 1) == 0,
                "failed to configure the bounded resident-cache test budget");
        require(dsmvc::metal::available(),
                "direct Metal scheduler test requires unified memory");
        auto first = dsmvc::metal::make_client();
        auto second = dsmvc::metal::make_client();

        constexpr std::uint32_t small_height = 24U;
        const auto small_plan = make_plan(64, 56);
        std::vector<float> small_source(
            static_cast<std::size_t>(small_plan->source_size) * small_height);
        for (std::size_t index = 0; index < small_source.size(); ++index) {
            small_source[index] = static_cast<float>((index * 37U + 11U) & 4095U)
                / 4095.0F;
        }
        test_shared_input_automatic_admission(
            small_plan, small_source, small_height);
        test_float64_plan_cpu_fallback(first);
        test_automatic_admission_boundaries(
            first, small_plan, small_source, small_height);
        test_recent_input_expiry_and_key_isolation(
            small_plan, small_source, small_height);
        test_shared_source_and_recovery(
            first, second, small_plan, small_source, small_height);

        std::vector<float> second_small_source = small_source;
        for (float &sample : second_small_source) sample *= 0.75F;
        test_interleaved_plan_offsets(
            first, small_plan, small_source, second_small_source, small_height);
        test_heterogeneous_plan_geometry(
            first, small_plan, small_source, small_height);
        test_heterogeneous_two_axis_geometry(
            first, small_plan, small_source, small_height);

        const auto horizontal = make_plan(384, 352);
        const auto vertical = make_plan(216, 200);
        std::vector<float> stress_source(
            static_cast<std::size_t>(horizontal->source_size)
            * vertical->source_size);
        for (std::size_t index = 0; index < stress_source.size(); ++index) {
            stress_source[index] = static_cast<float>((index * 19U + 7U) & 1023U)
                / 1023.0F;
        }
        test_ring_backpressure(first, horizontal, vertical, stress_source);
        test_error_propagation(
            first, small_plan, small_source, small_height);

        // The failed submission must not poison subsequent command buffers.
        test_shared_source_and_recovery(
            first, second, small_plan, small_source, small_height);

        test_resident_float_reuse_and_lifetime_identity(
            small_plan, small_height);
        test_resident_ready_singleton_after_recent_input_expiry(
            small_plan, small_height);
        test_resident_producer_failure_recovery(
            small_plan, small_height);
        const auto integer_vertical = make_plan(
            static_cast<std::int32_t>(small_height), 20);
        test_resident_integer_conversion_keys<std::uint8_t>(
            small_plan, integer_vertical,
            dsmvc::IntegerConversion{16.0F, 1.0F / 219.0F,
                                     219.0F, 16.0F, 255U},
            dsmvc::IntegerConversion{0.0F, 1.0F / 255.0F,
                                     255.0F, 0.0F, 255U});
        test_resident_integer_conversion_keys<std::uint16_t>(
            small_plan, integer_vertical,
            dsmvc::IntegerConversion{64.0F, 1.0F / 876.0F,
                                     876.0F, 64.0F, 1023U},
            dsmvc::IntegerConversion{0.0F, 1.0F / 1023.0F,
                                     1023.0F, 0.0F, 1023U});
        test_resident_integer_conversion_keys<std::uint16_t>(
            small_plan, integer_vertical,
            dsmvc::IntegerConversion{4096.0F, 1.0F / 56064.0F,
                                     56064.0F, 4096.0F, 65535U},
            dsmvc::IntegerConversion{0.0F, 1.0F / 65535.0F,
                                     65535.0F, 0.0F, 65535U});
        test_resident_in_flight_eviction_safety();

        first->close();
        std::vector<float> closed_output(
            static_cast<std::size_t>(small_plan->destination_size) * small_height);
        bool rejected = false;
        try {
            (void)dsmvc::metal::run(
                first,
                horizontal_job(
                    small_plan, small_source.data(), closed_output.data(),
                    small_height),
                [] {}, false);
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        require(rejected, "closed Metal client accepted a new request");
        second->close();

        const auto diagnostics = dsmvc::metal::diagnostics();
        std::cout << "Metal scheduler tests passed: submissions="
                  << diagnostics.submissions
                  << " max_in_flight=" << diagnostics.maximum_in_flight
                  << " plan_entries=" << diagnostics.plan_cache_entries
                  << " plan_evictions=" << diagnostics.plan_cache_evictions
                  << " resident_entries="
                  << diagnostics.resident_cache_entries
                  << " resident_evictions="
                  << diagnostics.resident_cache_evictions
                  << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Metal scheduler tests failed: " << error.what() << '\n';
        return 1;
    }
}
