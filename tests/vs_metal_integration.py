#!/usr/bin/env python3
"""Plugin-level correctness and lifetime checks for the Apple UMA Metal path."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import vapoursynth as vs


GEOMETRY = {
    "width": 1692,
    "height": 952,
    "src_left": 0.2222222222221717,
    "src_top": 0.25,
    "src_width": 1691.5555555555557,
    "src_height": 951.5,
}

CASES = {
    "bilinear": ("Debilinear", {}, 4),
    "spline16": ("Despline16", {}, 4),
    "bicubic": ("Debicubic", {"b": 0.0, "c": 0.5}, 4),
    "spline36": ("Despline36", {}, 7),
    "lanczos3": ("Delanczos", {"taps": 3}, 7),
    "spline64": ("Despline64", {}, 7),
}

GEOMETRY_SIGNATURE = "src:vnode;width:int;height:int;"
TAIL_SIGNATURE = (
    "src_left:float:opt;src_top:float:opt;"
    "src_width:float:opt;src_height:float:opt;"
    "border_handling:int:opt;force:int:opt;force_h:int:opt;"
    "force_v:int:opt;opt:int:opt;backend:data:opt;"
    "padding:int:opt;f64mode:int:opt;"
)
EXPECTED_SIGNATURES = {
    "Debilinear": GEOMETRY_SIGNATURE + TAIL_SIGNATURE,
    "Debicubic": GEOMETRY_SIGNATURE + "b:float:opt;c:float:opt;" + TAIL_SIGNATURE,
    "Delanczos": GEOMETRY_SIGNATURE + "taps:int:opt;" + TAIL_SIGNATURE,
    "Despline16": GEOMETRY_SIGNATURE + TAIL_SIGNATURE,
    "Despline36": GEOMETRY_SIGNATURE + TAIL_SIGNATURE,
    "Despline64": GEOMETRY_SIGNATURE + TAIL_SIGNATURE,
    "Descale": (
        GEOMETRY_SIGNATURE
        + "kernel:data:opt;taps:int:opt;b:float:opt;c:float:opt;"
        + "src_left:float:opt;src_top:float:opt;"
        + "src_width:float:opt;src_height:float:opt;"
        + "border_handling:int:opt;force:int:opt;force_h:int:opt;"
        + "force_v:int:opt;opt:int:opt;"
        + "custom:func:opt;support:int:opt;custom_kernel:func:opt;"
        + "backend:data:opt;padding:int:opt;f64mode:int:opt;"
    ),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def patterned_source(core: vs.Core, format_id: int,
                     length: int) -> vs.VideoNode:
    """Build two cached spatial patterns and alternate their range metadata."""

    def make_frame(range_value: int, seed: int) -> vs.VideoNode:
        blank = core.std.BlankClip(
            width=1920, height=1080, length=1, format=format_id)

        def fill(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
            del n
            output = f.copy()
            maximum = (1 << output.format.bits_per_sample) - 1
            for plane_index in range(output.format.num_planes):
                plane = np.asarray(output[plane_index])
                rows = np.arange(plane.shape[0], dtype=np.uint32)[:, None]
                columns = np.arange(plane.shape[1], dtype=np.uint32)[None, :]
                values = (
                    columns * 193
                    + rows * 389
                    + columns * rows * 17
                    + plane_index * 521
                    + seed * 977
                ) & maximum
                np.copyto(plane, values, casting="unsafe")
            output.props["_Range"] = range_value
            return output

        return core.std.ModifyFrame(blank, blank, fill)

    limited = make_frame(0, 1)
    full = make_frame(1, 2)
    clips = [limited, full] * ((length + 1) // 2)
    return core.std.Splice(clips)[:length]


def filtered(source: vs.VideoNode, case_name: str,
             backend: str) -> vs.VideoNode:
    function_name, case_arguments, _gpu_frames = CASES[case_name]
    return getattr(vs.core.dsmvc, function_name)(
        source, backend=backend, **GEOMETRY, **case_arguments)


def collect(clip: vs.VideoNode, prefetch: int) -> list[vs.VideoFrame]:
    return list(clip.frames(prefetch=prefetch, backlog=max(prefetch * 2, 2)))


def compare_frames(reference: list[vs.VideoFrame],
                   candidate: list[vs.VideoFrame], label: str) -> int:
    require(len(reference) == len(candidate), f"{label}: frame count differs")
    maximum = 0
    for frame_number, (cpu_frame, metal_frame) in enumerate(
            zip(reference, candidate, strict=True)):
        require(int(cpu_frame.props["_Range"]) == frame_number % 2,
                f"{label}: CPU range property differs at frame {frame_number}")
        require(int(metal_frame.props["_Range"]) == frame_number % 2,
                f"{label}: Metal range property differs at frame {frame_number}")
        for plane_index in range(cpu_frame.format.num_planes):
            cpu_plane = np.asarray(cpu_frame[plane_index]).astype(
                np.int32, copy=False)
            metal_plane = np.asarray(metal_frame[plane_index]).astype(
                np.int32, copy=False)
            require(cpu_plane.shape == metal_plane.shape,
                    f"{label}: plane {plane_index} shape differs")
            difference = int(np.max(np.abs(cpu_plane - metal_plane)))
            maximum = max(maximum, difference)
    require(maximum <= 1, f"{label}: maximum error is {maximum}, expected <= 1")
    return maximum


def metal_assignments(frames: list[vs.VideoFrame], expected_batch: int,
                      label: str, *,
                      track_range: bool = True,
                      copies_per_frame: int = 6) -> tuple[int, set[int]]:
    assigned = []
    ranges = set()
    for frame in frames:
        batch = int(frame.props.get("_DSMVCMetalBatch", -1))
        marker = int(frame.props.get("_DSMVCMetal", -1))
        staging_copies = int(
            frame.props.get("_DSMVCMetalStagingCopies", -1))
        staging_bytes = int(
            frame.props.get("_DSMVCMetalStagingBytes", -1))
        unique_inputs = int(
            frame.props.get("_DSMVCMetalUniqueInputs", -1))
        resident_producers = int(
            frame.props.get("_DSMVCMetalResidentProducers", -1))
        resident_hits = int(frame.props.get("_DSMVCMetalResidentHits", -1))
        resident_evictions = int(
            frame.props.get("_DSMVCMetalResidentEvictions", -1))
        resident_bytes = int(
            frame.props.get("_DSMVCMetalResidentBytes", -1))
        eliminated_staging_bytes = int(
            frame.props.get("_DSMVCMetalEliminatedStagingBytes", -1))
        gpu_interval_ns = int(
            frame.props.get("_DSMVCMetalGpuIntervalNs", -1))
        submission_gap_ns = int(
            frame.props.get("_DSMVCMetalSubmissionGapNs", -1))
        packing_metrics = [
            int(frame.props.get(name, -1))
            for name in (
                "_DSMVCCpuPlanPackExecutions",
                "_DSMVCCpuPlanPackWaits",
                "_DSMVCCpuPlanPackWaitNs",
                "_DSMVCCpuPlanLazyRequests",
                "_DSMVCCpuPlanLazyHits",
                "_DSMVCCpuPlanMaxConcurrentPacks",
            )
        ]
        require(marker == (1 if batch else 0),
                f"{label}: diagnostic properties disagree")
        require(all(metric >= 0 for metric in packing_metrics),
                f"{label}: CPU plan packing metrics are missing")
        if batch:
            minimum_batch = 1 if resident_hits > 0 else 2
            require(minimum_batch <= batch <= expected_batch,
                    f"{label}: Metal batch {batch}, expected "
                    f"{minimum_batch}..{expected_batch}")
            output_planes = copies_per_frame // 2
            require(1 <= unique_inputs <= expected_batch * output_planes,
                    f"{label}: invalid unique input count {unique_inputs}")
            require(resident_producers >= 0 and resident_hits >= 0
                    and resident_evictions >= 0 and resident_bytes >= 0
                    and eliminated_staging_bytes >= 0
                    and gpu_interval_ns >= 0 and submission_gap_ns >= 0,
                    f"{label}: resident diagnostics are missing")
            if resident_producers or resident_hits:
                require(batch * output_planes <= staging_copies
                        <= unique_inputs + batch * output_planes,
                        f"{label}: resident staging used "
                        f"{staging_copies} memcpy calls")
                require(resident_bytes > 0,
                        f"{label}: resident cache byte count is missing")
                if resident_hits:
                    require(eliminated_staging_bytes > 0,
                            f"{label}: resident hits eliminated no traffic")
            else:
                require(staging_copies
                        == unique_inputs + batch * output_planes,
                        f"{label}: staging used {staging_copies} memcpy calls")
            require(staging_bytes > 0,
                    f"{label}: staging byte count is missing")
            assigned.append(frame)
            if track_range:
                ranges.add(int(frame.props["_Range"]))
        else:
            require(staging_copies == 0 and staging_bytes == 0
                    and unique_inputs == 0 and resident_producers == 0
                    and resident_hits == 0 and resident_evictions == 0
                    and resident_bytes == 0 and eliminated_staging_bytes == 0
                    and gpu_interval_ns == 0 and submission_gap_ns == 0,
                    f"{label}: CPU frame reports Metal staging work")
    return len(assigned), ranges


def patterned_grays_source(core: vs.Core, length: int) -> vs.VideoNode:
    blank = core.std.BlankClip(
        width=1920, height=1080, length=1, format=vs.GRAYS, color=0.25)

    def fill(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        del n
        output = f.copy()
        plane = np.asarray(output[0])
        rows = np.arange(plane.shape[0], dtype=np.uint32)[:, None]
        columns = np.arange(plane.shape[1], dtype=np.uint32)[None, :]
        values = ((columns * 193 + rows * 389 + columns * rows * 17)
                  & 1023).astype(np.float32) / np.float32(1023.0)
        np.copyto(plane, values)
        return output

    pattern = core.std.ModifyFrame(blank, blank, fill)
    return core.std.Loop(pattern, times=length)


def compare_grays(reference: vs.VideoFrame, candidate: vs.VideoFrame,
                  label: str) -> float:
    difference = float(np.max(np.abs(
        np.asarray(reference[0]) - np.asarray(candidate[0]))))
    require(difference <= 3.0e-6,
            f"{label}: maximum error is {difference}, expected <= 3e-6")
    return difference


def test_conditioned_float64_fallback(core: vs.Core) -> None:
    source = patterned_grays_source(core, 1)
    geometry = {
        "width": 1920,
        "height": 980,
        "src_width": 1920.0,
        "src_height": 978.1,
        "src_top": 0.95,
        "force_h": 1,
    }
    for f64mode, label in ((0, "automatic"), (2, "forced")):
        cpu_frame = core.dsmvc.Delanczos(
            source, taps=2, backend="cpu", f64mode=f64mode,
            **geometry).get_frame(0)
        metal_frame = core.dsmvc.Delanczos(
            source, taps=2, backend="metal", f64mode=f64mode,
            **geometry).get_frame(0)
        case = f"conditioned-float64/{label}/explicit-metal"
        maximum = compare_grays(cpu_frame, metal_frame, case)
        assigned, _ranges = metal_assignments(
            [metal_frame], 1, case, track_range=False, copies_per_frame=2)
        require(assigned == 0,
                f"{label} Float64 plan entered the Metal executor")
        require(int(metal_frame.props.get("_DSMVCMetal", -1)) == 0,
                f"{label} Float64 plan did not publish _DSMVCMetal=0")
        print(f"{label} Float64 Metal fallback: max_error={maximum}")


def patterned_general_source(core: vs.Core, format_id: int,
                             length: int = 16, *, width: int = 128,
                             height: int = 96) -> vs.VideoNode:
    blank = core.std.BlankClip(
        width=width, height=height, length=1, format=format_id)

    def make(seed: int, range_value: int) -> vs.VideoNode:
        def fill(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
            del n
            output = f.copy()
            for plane_index in range(output.format.num_planes):
                plane = np.asarray(output[plane_index])
                rows = np.arange(plane.shape[0], dtype=np.uint32)[:, None]
                columns = np.arange(plane.shape[1], dtype=np.uint32)[None, :]
                values = columns * 37 + rows * 73 + columns * rows * 3
                values += plane_index * 101 + seed * 211
                if output.format.sample_type == vs.FLOAT:
                    normalized = (values & 4095).astype(np.float32) / np.float32(4095.0)
                    np.copyto(plane, normalized)
                else:
                    maximum = (1 << output.format.bits_per_sample) - 1
                    np.copyto(plane, values & maximum, casting="unsafe")
            output.props["_Range"] = range_value
            return output

        return core.std.ModifyFrame(blank, blank, fill)

    clips = [make(1, 0), make(2, 1)] * ((length + 1) // 2)
    return core.std.Splice(clips)[:length]


def compare_general_clips(core: vs.Core, cpu_clip: vs.VideoNode,
                          metal_clip: vs.VideoNode, label: str,
                          threads: int, *, expect_metal: bool = True) -> None:
    cpu_frames = collect(cpu_clip, threads)
    metal_frames = collect(metal_clip, threads)
    require(len(cpu_frames) == len(metal_frames), f"{label}: frame count differs")
    maximum = 0.0
    for cpu_frame, metal_frame in zip(cpu_frames, metal_frames, strict=True):
        require(cpu_frame.format.id == metal_frame.format.id,
                f"{label}: format differs")
        for plane_index in range(cpu_frame.format.num_planes):
            reference = np.asarray(cpu_frame[plane_index])
            candidate = np.asarray(metal_frame[plane_index])
            if cpu_frame.format.sample_type == vs.FLOAT:
                difference = float(np.max(np.abs(reference - candidate)))
            else:
                difference = float(np.max(np.abs(
                    reference.astype(np.int32) - candidate.astype(np.int32))))
            maximum = max(maximum, difference)
    tolerance = 3.0e-6 if cpu_clip.format.sample_type == vs.FLOAT else 1.0
    require(maximum <= tolerance,
            f"{label}: maximum error {maximum} exceeds {tolerance}")
    assigned, _ranges = metal_assignments(
        metal_frames, 7, label,
        copies_per_frame=2 * metal_clip.format.num_planes)
    if expect_metal:
        require(assigned > 0, f"{label}: no frame executed on Metal")
    else:
        require(assigned == 0,
                f"{label}: conditioned plan entered the Metal executor")


def test_general_surface(core: vs.Core, threads: int) -> None:
    geometry = {
        "width": 112, "height": 80,
        "src_left": 0.125, "src_top": 0.375,
        "src_width": 111.75, "src_height": 79.5,
    }
    for format_id in (
            vs.GRAY16, vs.RGB24, vs.YUV422P12, vs.YUV444P16,
            vs.RGBS, vs.YUV444PS):
        source = patterned_general_source(core, format_id)
        cpu = core.dsmvc.Debicubic(
            source, backend="cpu", border_handling=1, **geometry)
        metal = core.dsmvc.Debicubic(
            source, backend="metal", border_handling=1, **geometry)
        compare_general_clips(
            core, cpu, metal, f"general/{source.format.name}", threads)

    reproduced_source = patterned_general_source(
        core, vs.GRAY16, width=96, height=64)
    reproduced_geometry = {"width": 80, "height": 48}
    compare_general_clips(
        core,
        core.dsmvc.Debicubic(
            reproduced_source, backend="cpu", opt=1, f64mode=1,
            **reproduced_geometry),
        core.dsmvc.Debicubic(
            reproduced_source, backend="metal", f64mode=1,
            **reproduced_geometry),
        "integer-contract/GRAY16-96x64-to-80x48", threads)

    vertical_source = patterned_general_source(core, vs.YUV444PS)
    compare_general_clips(
        core,
        core.dsmvc.Despline36(
            vertical_source, width=128, height=80, src_top=0.25,
            src_height=79.5, backend="cpu"),
        core.dsmvc.Despline36(
            vertical_source, width=128, height=80, src_top=0.25,
            src_height=79.5, backend="metal"),
        "general/vertical-only", threads)

    custom_source = patterned_general_source(core, vs.GRAYS)
    custom_kernel = lambda x: max(1.0 - abs(x), 0.0)
    compare_general_clips(
        core,
        core.dsmvc.Descale(
            custom_source, width=112, height=96, src_left=0.125,
            src_width=111.75, custom_kernel=custom_kernel, taps=1,
            border_handling=0, backend="cpu"),
        core.dsmvc.Descale(
            custom_source, width=112, height=96, src_left=0.125,
            src_width=111.75, custom_kernel=custom_kernel, taps=1,
            border_handling=0, backend="metal"),
        "general/horizontal-custom", threads)

    conditioned_custom = lambda x: max(1.0 - abs(x) / 5.0, 0.0)
    compare_general_clips(
        core,
        core.dsmvc.Descale(
            custom_source, width=112, height=96, src_left=0.125,
            src_width=111.75, custom_kernel=conditioned_custom, taps=5,
            border_handling=0, backend="cpu"),
        core.dsmvc.Descale(
            custom_source, width=112, height=96, src_left=0.125,
            src_width=111.75, custom_kernel=conditioned_custom, taps=5,
            border_handling=0, backend="metal"),
        "general/horizontal-custom-conditioned", threads,
        expect_metal=False)

    generic_source = patterned_general_source(core, vs.GRAYS)
    compare_general_clips(
        core,
        core.dsmvc.Delanczos(
            generic_source, taps=9, backend="cpu", **geometry),
        core.dsmvc.Delanczos(
            generic_source, taps=9, backend="metal", **geometry),
        "general/generic-bandwidth", threads)


def compare_case(core: vs.Core, format_id: int, case_name: str,
                 threads: int, length: int = 16) -> tuple[int, set[int], vs.VideoFrame]:
    source = patterned_source(core, format_id, length)
    cpu_clip = filtered(source, case_name, "cpu")
    metal_clip = filtered(source, case_name, "metal")
    cpu_frames = collect(cpu_clip, threads)
    metal_frames = collect(metal_clip, threads)
    maximum = compare_frames(
        cpu_frames, metal_frames,
        f"{source.format.name}/{case_name}/length-{length}")
    expected_batch = CASES[case_name][2]
    assigned, ranges = metal_assignments(
        metal_frames, expected_batch,
        f"{source.format.name}/{case_name}/length-{length}")
    require(0 < assigned <= length,
            f"{source.format.name}/{case_name}/length-{length}: "
            f"Metal assigned {assigned} frames")
    retained = next(
        frame for frame in metal_frames
        if int(frame.props["_DSMVCMetalBatch"]) != 0)
    del cpu_frames, metal_frames, cpu_clip, metal_clip, source
    gc.collect()
    return maximum, ranges, retained


def expect_error(callback, contains: str) -> None:
    try:
        callback()
    except vs.Error as error:
        require(contains.lower() in str(error).lower(),
                f"unexpected error: {error}")
        return
    raise AssertionError(f"expected a VapourSynth error containing {contains!r}")


def test_precision_and_padding_arguments(
        core: vs.Core, source: vs.VideoNode) -> None:
    geometry = {
        "width": 80, "height": 48,
        "src_left": 0.25, "src_top": 0.125,
        "src_width": 79.5, "src_height": 47.75,
    }
    default = core.dsmvc.Debicubic(
        source, backend="cpu", **geometry).get_frame(0)
    symmetric = core.dsmvc.Debicubic(
        source, backend="cpu", padding=3, **geometry).get_frame(0)
    require(np.array_equal(np.asarray(default[0]), np.asarray(symmetric[0])),
            "default padding is not symmetric")
    for padding in range(4):
        core.dsmvc.Debicubic(
            source, backend="cpu", padding=padding,
            **geometry).get_frame(0)

    expect_error(
        lambda: core.dsmvc.Debicubic(
            source, backend="cpu", padding=3, border_handling=0,
            **geometry),
        "either padding or border_handling")
    expect_error(
        lambda: core.dsmvc.Debicubic(
            source, backend="cpu", padding=4, **geometry),
        "padding must be")
    expect_error(
        lambda: core.dsmvc.Debicubic(
            source, backend="cpu", f64mode=3, **geometry),
        "f64mode must be")

    automatic = core.dsmvc.Debicubic(
        source, backend="cpu", f64mode=0, **geometry).get_frame(0)
    float64 = core.dsmvc.Debicubic(
        source, backend="cpu", f64mode=2, **geometry).get_frame(0)
    maximum = float(np.max(np.abs(
        np.asarray(automatic[0]) - np.asarray(float64[0]))))
    require(maximum <= 2.0e-5,
            f"forced Float64 CPU output differs from automatic: {maximum}")


def test_tails(core: vs.Core, threads: int) -> None:
    for length in (7, 17, 32):
        source = patterned_source(core, vs.YUV420P8, length)
        cpu_clip = filtered(source, "spline64", "cpu")
        metal_clip = filtered(source, "spline64", "metal")
        cpu_frames = collect(cpu_clip, threads)
        metal_frames = collect(metal_clip, threads)
        compare_frames(cpu_frames, metal_frames, f"tail/length-{length}")
        assigned, _ranges = metal_assignments(
            metal_frames, 7, f"tail/length-{length}")
        require(assigned <= length,
                f"tail/length-{length}: assigned {assigned} Metal frames")
        del cpu_frames, metal_frames, cpu_clip, metal_clip, source
        gc.collect()


def test_cancellation(core: vs.Core, threads: int) -> None:
    source = patterned_source(core, vs.YUV420P8, 64)
    clip = filtered(source, "spline64", "metal")
    iterator = clip.frames(prefetch=threads, backlog=threads * 2)
    first = next(iterator)
    require(first.width == GEOMETRY["width"],
            "cancellation probe did not yield a valid frame")
    iterator.close()
    del first, iterator, clip, source
    gc.collect()

    probe = filtered(patterned_source(core, vs.YUV420P8, 3),
                     "spline64", "metal")
    frames = collect(probe, threads)
    assigned, _ranges = metal_assignments(frames, 7, "post-cancel probe")
    require(assigned <= 3, "post-cancel probe reported impossible assignments")


def test_auto_admission(core: vs.Core, threads: int) -> None:
    source = patterned_source(core, vs.YUV420P8, 64)
    cpu_clip = filtered(source, "spline64", "cpu")
    auto_clip = filtered(source, "spline64", "auto")
    cpu_frames = collect(cpu_clip, threads)
    auto_frames = collect(auto_clip, threads)
    compare_frames(cpu_frames, auto_frames, "auto/wide/high-concurrency")
    assigned, ranges = metal_assignments(
        auto_frames, 7, "auto/wide/high-concurrency")
    require(assigned > 0, "eligible auto route never activated Metal")
    require(ranges == {0, 1},
            "eligible auto route did not cover both range modes")
    del cpu_frames, auto_frames, cpu_clip, auto_clip, source
    gc.collect()

    low_source = patterned_source(core, vs.YUV420P8, 64)
    low_clip = filtered(low_source, "spline64", "auto")
    low_frames = collect(low_clip, 4)
    assigned, _ranges = metal_assignments(
        low_frames, 7, "auto/wide/low-concurrency")
    require(assigned == 0,
            "low-concurrency auto route unexpectedly activated Metal")
    del low_frames, low_clip, low_source
    gc.collect()

    narrow_source = patterned_source(core, vs.YUV420P8, 64)
    narrow_frame = filtered(
        narrow_source, "bilinear", "auto").get_frame(0)
    require(int(narrow_frame.props.get("_DSMVCMetal", -1)) == 0,
            "narrow auto route executed on Metal")
    del narrow_frame, narrow_source
    gc.collect()


def test_grays_b1_metal(core: vs.Core, threads: int) -> None:
    explicit_source = patterned_grays_source(core, 16)
    cpu_reference = filtered(
        explicit_source, "bilinear", "cpu").get_frame(0)
    explicit_frames = collect(
        filtered(explicit_source, "bilinear", "metal"), threads)
    maximum = 0.0
    for frame in explicit_frames:
        maximum = max(maximum, compare_grays(
            cpu_reference, frame, "GRAYS/B1/explicit-metal"))
    assigned, _ranges = metal_assignments(
        explicit_frames, 4, "GRAYS/B1/explicit-metal", track_range=False,
        copies_per_frame=2)
    require(0 < assigned <= 4,
            f"GRAYS/B1/explicit-metal: assigned {assigned} Metal frames")
    del explicit_frames, explicit_source
    gc.collect()

    wide_auto_source = patterned_grays_source(core, 64)
    wide_cpu_reference = filtered(
        wide_auto_source, "spline64", "cpu").get_frame(0)
    wide_auto_frames = collect(
        filtered(wide_auto_source, "spline64", "auto"), threads)
    for frame in wide_auto_frames:
        maximum = max(maximum, compare_grays(
            wide_cpu_reference, frame, "GRAYS/B7/automatic"))
    wide_assigned, _ranges = metal_assignments(
        wide_auto_frames, 7, "GRAYS/B7/automatic", track_range=False,
        copies_per_frame=2)
    require(wide_assigned > 0,
            "eligible GRAYS automatic route never activated Metal")
    del wide_auto_frames, wide_cpu_reference, wide_auto_source
    gc.collect()

    narrow_auto_source = patterned_grays_source(core, 64)
    auto_frame = filtered(narrow_auto_source, "bilinear", "auto").get_frame(0)
    require(int(auto_frame.props.get("_DSMVCMetal", -1)) == 0,
            "GRAYS B1 auto route bypassed its measured CPU fallback")
    del auto_frame, narrow_auto_source
    gc.collect()
    print(f"GRAYS explicit/automatic Metal: max_error={maximum}")


def run(options: argparse.Namespace) -> None:
    core = vs.core
    core.num_threads = options.threads
    core.std.LoadPlugin(path=str(Path(options.plugin).resolve()))

    matches = [plugin for plugin in core.plugins()
               if plugin.identifier == "com.dsmvc.descale"]
    require(len(matches) == 1, "new plugin ID was not registered exactly once")
    require(matches[0].namespace == "dsmvc", "plugin namespace is not dsmvc")
    require({function.name for function in matches[0].functions()}
            == set(EXPECTED_SIGNATURES), "public function set differs")
    for name, signature in EXPECTED_SIGNATURES.items():
        function = getattr(core.dsmvc, name)
        require(function.signature == signature,
                f"{name}: API4 input signature differs")
        require(function.return_signature == "clip:vnode;",
                f"{name}: API4 return signature differs")

    cpu_source = core.std.BlankClip(
        width=96, height=64, length=1, format=vs.GRAYS, color=0.35)
    scalar = core.dsmvc.Debicubic(
        cpu_source, width=80, height=48, opt=1, backend="cpu").get_frame(0)
    simd = core.dsmvc.Debicubic(
        cpu_source, width=80, height=48, opt=2, backend="cpu").get_frame(0)
    cpu_error = float(np.max(np.abs(
        np.asarray(scalar[0]) - np.asarray(simd[0]))))
    require(cpu_error <= 1.5e-6,
            f"CPU scalar/native SIMD mismatch: {cpu_error}")
    test_precision_and_padding_arguments(core, cpu_source)
    for backend, enabled in (
            ("vulkan", options.vulkan_enabled),
            ("cuda", options.cuda_enabled)):
        if not enabled:
            expect_error(
                lambda backend=backend: core.dsmvc.Debicubic(
                    cpu_source, width=80, height=48, backend=backend),
                "not compiled")

    if options.expect_metal == "disabled":
        expect_error(
            lambda: core.dsmvc.Debicubic(
                cpu_source, width=80, height=48, backend="metal"),
            "not compiled")
        print(
            "dsmvc API4 Metal-off smoke passed: "
            f"scalar/native SIMD max_error={cpu_error}")
        return

    test_conditioned_float64_fallback(core)
    if options.conditioned_only:
        print("dsmvc conditioned Float64 Metal fallback test passed")
        return

    all_gpu_ranges = {vs.YUV420P8: set(), vs.YUV420P10: set()}
    retained_frame = None
    for format_id in (vs.YUV420P8, vs.YUV420P10):
        for case_name in CASES:
            maximum, gpu_ranges, retained = compare_case(
                core, format_id, case_name, options.threads)
            all_gpu_ranges[format_id].update(gpu_ranges)
            retained_frame = retained
            print(
                f"{retained.format.name}/{case_name}: "
                f"max_error={maximum} gpu_ranges={sorted(gpu_ranges)}")
    for format_id, ranges in all_gpu_ranges.items():
        require(ranges == {0, 1},
                f"format {format_id}: both range modes were not assigned to Metal")

    require(retained_frame is not None, "no Metal frame was retained")
    retained_value = int(np.asarray(retained_frame[0])[0, 0])
    retained_batch = int(retained_frame.props["_DSMVCMetalBatch"])
    gc.collect()
    require(int(np.asarray(retained_frame[0])[0, 0]) == retained_value,
            "retained Metal frame data changed after graph release")
    require(int(retained_frame.props["_DSMVCMetalBatch"]) == retained_batch,
            "retained Metal frame properties changed after graph release")

    auto_source = patterned_source(core, vs.YUV420P8, 1)
    auto_frame = filtered(auto_source, "bilinear", "auto").get_frame(0)
    require("_DSMVCMetal" not in auto_frame.props,
            "backend=auto unexpectedly entered the Metal path")

    general_source = patterned_source(core, vs.YUV420P8, 16)
    general_geometry = {
        "width": 1600, "height": 900,
        "src_left": 0.125, "src_top": 0.375,
        "src_width": 1599.75, "src_height": 899.5,
    }
    general_cpu = collect(core.dsmvc.Debilinear(
        general_source, backend="cpu", **general_geometry), options.threads)
    general_metal = collect(core.dsmvc.Debilinear(
        general_source, backend="metal", **general_geometry), options.threads)
    compare_frames(general_cpu, general_metal, "general-geometry")
    assigned, _ranges = metal_assignments(
        general_metal, 4, "general-geometry")
    require(0 < assigned <= 16,
            f"general-geometry: assigned {assigned} Metal frames")

    test_tails(core, options.threads)
    test_cancellation(core, options.threads)
    test_auto_admission(core, options.threads)
    test_grays_b1_metal(core, options.threads)
    test_general_surface(core, options.threads)
    print("dsmvc Apple UMA Metal VapourSynth integration tests passed")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--plugin", required=True)
    result.add_argument("--threads", type=int, default=16)
    result.add_argument(
        "--expect-metal", choices=("enabled", "disabled"),
        default="enabled")
    result.add_argument("--conditioned-only", action="store_true")
    result.add_argument("--cuda-enabled", action="store_true")
    result.add_argument("--vulkan-enabled", action="store_true")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
