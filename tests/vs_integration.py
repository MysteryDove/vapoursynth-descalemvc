#!/usr/bin/env python3
"""VapourSynth API4 integration and baseline compatibility checks."""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import vapoursynth as vs


FUNCTIONS = {
    "Debilinear": {},
    "Debicubic": {"b": 0.0, "c": 1.0},
    "Delanczos": {"taps": 3},
    "Despline16": {},
    "Despline36": {},
    "Despline64": {},
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


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None,
            f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_clips(old, new, label: str) -> None:
    require((old.width, old.height, old.format.id)
            == (new.width, new.height, new.format.id),
            f"{label}: output shape or format differs")
    old_frame = old.get_frame(0)
    new_frame = new.get_frame(0)
    maximum = 0.0
    total = 0.0
    count = 0
    for plane in range(old.format.num_planes):
        old_view = memoryview(old_frame[plane])
        new_view = memoryview(new_frame[plane])
        require(old_view.shape == new_view.shape,
                f"{label}: plane {plane} shape differs")
        for y in range(old_view.shape[0]):
            for x in range(old_view.shape[1]):
                difference = abs(float(old_view[y, x]) - float(new_view[y, x]))
                maximum = max(maximum, difference)
                total += difference
                count += 1
    mean = total / count if count else 0.0
    if old.format.sample_type == vs.FLOAT:
        require(maximum <= 2.0e-5 and mean <= 1.0e-6,
                f"{label}: float mismatch max={maximum} mean={mean}")
    else:
        require(maximum <= 1.0,
                f"{label}: integer mismatch max={maximum}")


def expect_error(callback, contains: str) -> None:
    try:
        callback()
    except vs.Error as error:
        require(contains.lower() in str(error).lower(),
                f"unexpected error: {error}")
        return
    raise AssertionError(f"expected an error containing {contains!r}")


def direct_call(namespace, name: str, source, **overrides):
    arguments = dict(width=80, height=48)
    arguments.update(FUNCTIONS.get(name, {}))
    arguments.update(overrides)
    return getattr(namespace, name)(source, **arguments)


def baseline_call(namespace, name: str, source, **overrides):
    if source.format.sample_type == vs.FLOAT and source.format.bits_per_sample == 32:
        return direct_call(namespace, name, source, **overrides)
    float_format = vs.core.query_video_format(
        source.format.color_family, vs.FLOAT, 32,
        source.format.subsampling_w, source.format.subsampling_h)
    float_source = source.resize.Point(
        format=float_format.id, dither_type="none")
    float_output = direct_call(namespace, name, float_source, **overrides)
    return float_output.resize.Point(
        format=source.format.id, dither_type="none")


def patterned_integer_clip(core, format_id, property_name: str,
                           range_value: int, *, width: int = 96,
                           height: int = 64):
    blank = core.std.BlankClip(width=width, height=height, format=format_id)

    def fill(n, f):
        del n
        output = f.copy()
        maximum = (1 << output.format.bits_per_sample) - 1
        for plane_index in range(output.format.num_planes):
            plane = output[plane_index]
            for y in range(plane.shape[0]):
                for x in range(plane.shape[1]):
                    code = (x * 193 + y * 389 + plane_index * 521
                            + x * y * 17)
                    plane[y, x] = code % (maximum + 1)
        output.props[property_name] = range_value
        return output

    return core.std.ModifyFrame(blank, blank, fill)


def require_frames_bit_exact(reference, candidate, label: str) -> None:
    require(len(reference) == len(candidate), f"{label}: frame count differs")
    for frame_number, (left, right) in enumerate(
            zip(reference, candidate, strict=True)):
        require(left.format.id == right.format.id,
                f"{label}: format differs at frame {frame_number}")
        for plane in range(left.format.num_planes):
            left_view = memoryview(left[plane])
            right_view = memoryview(right[plane])
            require(left_view.shape == right_view.shape,
                    f"{label}: plane shape differs at frame {frame_number}")
            require(left_view.tobytes() == right_view.tobytes(),
                    f"{label}: output is not bit exact at frame "
                    f"{frame_number}, plane {plane}")


def test_cpu_dynamic_route(core, threads: int) -> None:
    require(threads > 1, "CPU dynamic-route test requires multiple core threads")
    frame_count = max(8, min(threads, 16))

    float_center = core.std.BlankClip(
        width=400, height=300, length=1, format=vs.GRAYS, color=[0.2])
    float_frame = core.std.AddBorders(
        float_center, left=200, right=200, top=150, bottom=150,
        color=[0.8])
    float_source = core.std.Loop(float_frame, times=frame_count)

    integer_frame = patterned_integer_clip(
        core, vs.GRAY16, "_Range", 1, width=800, height=600)
    integer_source = core.std.Loop(integer_frame, times=frame_count)

    for source, label in (
            (float_source, "float-two-pass-vs-fused"),
            (integer_source, "integer-buffered-vs-streamed")):
        serial_clip = core.dsmvc.Delanczos(
            source, width=640, height=480, taps=3, backend="cpu")
        concurrent_clip = core.dsmvc.Delanczos(
            source, width=640, height=480, taps=3, backend="cpu")
        serial_frames = [serial_clip.get_frame(n) for n in range(frame_count)]
        concurrent_frames = list(concurrent_clip.frames(
            prefetch=frame_count, backlog=frame_count * 2))
        require_frames_bit_exact(
            serial_frames, concurrent_frames, f"cpu-dynamic-route/{label}")


def run(options) -> None:
    core = vs.core
    core.num_threads = options.threads
    if not hasattr(core, "descale"):
        core.std.LoadPlugin(path=str(Path(options.old_plugin).resolve()))
    core.std.LoadPlugin(path=str(Path(options.plugin).resolve()))

    matches = [plugin for plugin in core.plugins()
               if plugin.identifier == "com.dsmvc.descale"]
    require(len(matches) == 1, "new plugin ID was not registered exactly once")
    require(matches[0].namespace == "dsmvc", "plugin namespace is not dsmvc")
    require({function.name for function in matches[0].functions()} ==
            set(FUNCTIONS) | {"Descale"}, "public function set differs")
    for name in set(FUNCTIONS) | {"Descale"}:
        function = getattr(core.dsmvc, name)
        require(function.signature == EXPECTED_SIGNATURES[name],
                f"{name}: API4 input signature differs")
        require(function.return_signature == "clip:vnode;",
                f"{name}: API4 return signature differs")

    float_source = core.std.BlankClip(
        width=96, height=64, format=vs.GRAYS, color=[0.35])
    for name in FUNCTIONS:
        old = direct_call(core.descale, name, float_source)
        new = direct_call(core.dsmvc, name, float_source, backend="cpu")
        compare_clips(old, new, f"function/{name}")

    formats = (
        vs.GRAY8, vs.GRAY16, vs.GRAYS,
        vs.RGB24, vs.RGBS,
        vs.YUV420P10, vs.YUV444PS,
    )
    for format_id in formats:
        source = core.std.BlankClip(width=96, height=64, format=format_id)
        old = baseline_call(core.descale, "Debicubic", source)
        new = direct_call(core.dsmvc, "Debicubic", source, backend="auto")
        compare_clips(old, new, f"format/{source.format.name}")

    for format_id in (vs.GRAY8, vs.GRAY16, vs.RGB24, vs.YUV420P10):
        for property_name, range_values in (
                ("_Range", (0, 1)),
                ("_ColorRange", (0, 1))):
            for range_value in range_values:
                source = patterned_integer_clip(
                    core, format_id, property_name, int(range_value))
                for name in ("Debilinear", "Delanczos", "Despline64"):
                    old = baseline_call(core.descale, name, source)
                    new = direct_call(
                        core.dsmvc, name, source, backend="cpu")
                    compare_clips(
                        old, new,
                        f"range/{source.format.name}/{property_name}/"
                        f"{range_value}/{name}")

    patterned_gray16 = patterned_integer_clip(
        core, vs.GRAY16, "_Range", 1)
    patterned_scalar = direct_call(
        core.dsmvc, "Debicubic", patterned_gray16,
        backend="cpu", opt=1, f64mode=1)
    patterned_simd = direct_call(
        core.dsmvc, "Debicubic", patterned_gray16,
        backend="cpu", opt=2, f64mode=1)
    compare_clips(
        patterned_scalar, patterned_simd,
        "integer-contract/GRAY16-96x64-to-80x48/scalar-vs-simd")

    geometry = {
        "src_left": 0.25,
        "src_top": 0.125,
        "src_width": 79.5,
        "src_height": 47.75,
    }
    for border in (0, 1, 2):
        old = direct_call(core.descale, "Debicubic", float_source,
                          border_handling=border, **geometry)
        new = direct_call(core.dsmvc, "Debicubic", float_source,
                          border_handling=border, backend="cpu", **geometry)
        compare_clips(old, new, f"border/{border}")

    default_padding = direct_call(
        core.dsmvc, "Debicubic", float_source, backend="cpu", **geometry)
    symmetric_padding = direct_call(
        core.dsmvc, "Debicubic", float_source,
        padding=3, backend="cpu", **geometry)
    compare_clips(default_padding, symmetric_padding, "padding/default-symmetric")
    for padding in range(4):
        direct_call(
            core.dsmvc, "Debicubic", float_source,
            padding=padding, backend="cpu", **geometry).get_frame(0)
    expect_error(
        lambda: direct_call(
            core.dsmvc, "Debicubic", float_source,
            padding=3, border_handling=0, backend="cpu", **geometry),
        "either padding or border_handling")
    expect_error(
        lambda: direct_call(
            core.dsmvc, "Debicubic", float_source,
            padding=4, backend="cpu", **geometry),
        "padding must be")
    expect_error(
        lambda: direct_call(
            core.dsmvc, "Debicubic", float_source,
            f64mode=3, backend="cpu", **geometry),
        "f64mode must be")

    automatic_precision = direct_call(
        core.dsmvc, "Debicubic", float_source,
        f64mode=0, backend="cpu", **geometry)
    forced_float64 = direct_call(
        core.dsmvc, "Debicubic", float_source,
        f64mode=2, backend="cpu", **geometry)
    compare_clips(automatic_precision, forced_float64, "f64mode/forced-f64")

    identity_arguments = {"width": 96, "height": 64,
                          "src_width": 96.0, "src_height": 64.0}
    for flags in ({"force": 1}, {"force_h": 1}, {"force_v": 1}):
        old = core.descale.Debicubic(float_source, **identity_arguments, **flags)
        new = core.dsmvc.Debicubic(
            float_source, **identity_arguments, backend="cpu", **flags)
        compare_clips(old, new, "force/" + next(iter(flags)))

    custom_source = core.std.AddBorders(
        core.std.BlankClip(width=64, height=32, format=vs.GRAYS, color=[0.2]),
        left=16, right=16, top=16, bottom=16, color=[0.8])
    custom_kernel = lambda x: max(1.0 - abs(x) / 3.0, 0.0)
    custom_alias = lambda x: max(1.0 - abs(x), 0.0)
    custom_cases = (
        ("custom-kernel", {"custom_kernel": custom_kernel, "taps": 2}),
        ("custom-support", {"custom": custom_kernel, "support": 2}),
        ("custom-precedence", {
            "custom": custom_alias, "custom_kernel": custom_kernel,
            "support": 2, "taps": 1,
        }),
    )
    old_descale = getattr(core.descale, "Descale", None)
    new_custom_outputs = {}
    for label, arguments in custom_cases:
        new_custom = core.dsmvc.Descale(
            custom_source, width=80, height=48, backend="cpu", **arguments)
        new_custom_outputs[label] = new_custom
        if old_descale is not None:
            old_custom = old_descale(
                custom_source, width=80, height=48, **arguments)
            compare_clips(old_custom, new_custom, label)
        else:
            new_custom.get_frame(0)
    if old_descale is None:
        compare_clips(
            new_custom_outputs["custom-kernel"],
            new_custom_outputs["custom-support"], "custom/aliases")
        precedence_reference = core.dsmvc.Descale(
            custom_source, width=80, height=48, custom=custom_alias,
            taps=1, backend="cpu")
        compare_clips(
            precedence_reference, new_custom_outputs["custom-precedence"],
            "custom/precedence")

    large_support_source = core.std.BlankClip(
        width=160, height=160, format=vs.GRAYS, color=[0.4])
    compare_clips(
        core.descale.Delanczos(
            large_support_source, width=128, height=128, taps=16),
        core.dsmvc.Delanczos(
            large_support_source, width=128, height=128, taps=16,
            backend="cpu"),
        "lanczos/taps-16")
    large_support_kernel = lambda x: max(1.0 - abs(x), 0.0)
    large_custom = core.dsmvc.Descale(
        large_support_source, width=128, height=128,
        custom_kernel=large_support_kernel, taps=65, backend="cpu")
    if old_descale is not None:
        compare_clips(
            old_descale(
                large_support_source, width=128, height=128,
                custom_kernel=large_support_kernel, taps=65),
            large_custom, "custom/taps-65")
    else:
        large_custom.get_frame(0)

    scalar = core.dsmvc.Debicubic(
        float_source, width=80, height=48, opt=1, backend="cpu")
    simd = core.dsmvc.Debicubic(
        float_source, width=80, height=48, opt=2, backend="cpu")
    compare_clips(scalar, simd, "opt/scalar-vs-simd")

    expect_error(
        lambda: core.dsmvc.Debicubic(
            float_source, width=80, height=48, backend="metal"),
        "backend 'metal' is not compiled")
    if options.vulkan_enabled:
        for name in FUNCTIONS:
            cpu = direct_call(core.dsmvc, name, float_source, backend="cpu")
            vulkan = direct_call(
                core.dsmvc, name, float_source, backend="vulkan")
            compare_clips(cpu, vulkan, f"backend/cpu-vs-vulkan/{name}")
        for format_id in formats:
            source = core.std.BlankClip(width=96, height=64, format=format_id)
            cpu = direct_call(
                core.dsmvc, "Debicubic", source, backend="cpu")
            vulkan = direct_call(
                core.dsmvc, "Debicubic", source, backend="vulkan")
            compare_clips(
                cpu, vulkan, f"backend/cpu-vs-vulkan/{source.format.name}")
        patterned_vulkan = direct_call(
            core.dsmvc, "Debicubic", patterned_gray16,
            backend="vulkan", f64mode=1)
        compare_clips(
            patterned_scalar, patterned_vulkan,
            "integer-contract/GRAY16-96x64-to-80x48/vulkan")
    else:
        expect_error(
            lambda: core.dsmvc.Debicubic(
                float_source, width=80, height=48, backend="vulkan"),
            "backend 'vulkan' is not compiled")
    if options.cuda_enabled:
        cuda = core.dsmvc.Debicubic(
            float_source, width=80, height=48, backend="cuda")
        compare_clips(scalar, cuda, "backend/cpu-vs-cuda")
        patterned_cuda = direct_call(
            core.dsmvc, "Debicubic", patterned_gray16,
            backend="cuda", f64mode=1)
        compare_clips(
            patterned_scalar, patterned_cuda,
            "integer-contract/GRAY16-96x64-to-80x48/cuda")
    else:
        expect_error(
            lambda: core.dsmvc.Debicubic(
                float_source, width=80, height=48, backend="cuda"),
            "backend 'cuda' is not compiled")
    expect_error(
        lambda: core.dsmvc.Debicubic(
            float_source, width=80, height=48, backend="invalid"),
        "backend must be")
    expect_error(
        lambda: core.dsmvc.Descale(float_source, width=80, height=48),
        "kernel or custom kernel is required")
    expect_error(
        lambda: core.dsmvc.Debicubic(float_source, width=0, height=48),
        "width must be greater than zero")
    expect_error(
        lambda: core.dsmvc.Debicubic(float_source, width=80, height=7),
        "height must be at least 8")
    subsampled = core.std.BlankClip(width=96, height=64, format=vs.YUV420P10)
    expect_error(
        lambda: core.dsmvc.Debicubic(subsampled, width=79, height=48),
        "incompatible with subsampling")

    new_wrapper = load_module(
        "dsmvc_test_new_wrapper", Path(options.repo_root) / "dsmvc.py")
    old_wrapper_path = (Path(options.old_wrapper).resolve()
                        if options.old_wrapper else
                        Path(options.vs_root) / "VapourSynthScripts" / "descale.py")
    old_wrapper = (
        load_module("dsmvc_test_old_wrapper", old_wrapper_path)
        if old_descale is not None and old_wrapper_path.is_file() else None)
    require(new_wrapper.Opt.SIMD == new_wrapper.Opt.AVX2
            == new_wrapper.Opt.NEON == 2,
            "wrapper SIMD option aliases differ")
    require(new_wrapper.Opt(2).name == "AVX2",
            "wrapper changed the canonical legacy AVX2 option name")
    require(tuple(int(value) for value in new_wrapper.Padding) == (0, 1, 2, 3),
            "wrapper padding values differ")
    require(tuple(int(value) for value in new_wrapper.F64Mode) == (0, 1, 2),
            "wrapper f64mode values differ")
    rgb = core.std.BlankClip(width=96, height=64, format=vs.RGB24)
    new_rgb = new_wrapper.Debicubic(
        rgb, 80, 48, b=0.0, c=1.0,
        opt=new_wrapper.Opt.NONE, backend="cpu")
    yuv = core.std.BlankClip(width=96, height=64, format=vs.YUV420P10)
    new_gray = new_wrapper.Debicubic(
        yuv, 80, 48, gray=True, backend="cpu")
    new_yuv444 = new_wrapper.Debicubic(
        yuv, 80, 48, yuv444=True, backend="cpu")
    new_explicit_modes = new_wrapper.Debicubic(
        yuv, 80, 48, gray=True, backend="cpu",
        padding=new_wrapper.Padding.REFLECT101,
        f64mode=new_wrapper.F64Mode.F64)
    new_explicit_modes.get_frame(0)
    try:
        new_wrapper.Debicubic(
            yuv, 80, 48, gray=True,
            border_handling=new_wrapper.BorderHandling.MIRROR,
            padding=new_wrapper.Padding.SYMMETRIC)
    except ValueError as error:
        require("either padding or border_handling" in str(error),
                f"unexpected wrapper conflict error: {error}")
    else:
        raise AssertionError("wrapper accepted padding with border_handling")
    if old_wrapper is not None:
        compare_clips(
            old_wrapper.Debicubic(rgb, 80, 48, b=0.0, c=1.0),
            new_rgb, "wrapper/RGB24")
        compare_clips(
            old_wrapper.Debicubic(yuv, 80, 48, gray=True),
            new_gray, "wrapper/gray")
        compare_clips(
            old_wrapper.Debicubic(yuv, 80, 48, yuv444=True),
            new_yuv444, "wrapper/yuv444")
    else:
        for clip in (new_rgb, new_gray, new_yuv444):
            require((clip.width, clip.height) == (80, 48),
                    "wrapper output dimensions differ")
            clip.get_frame(0)

    large_source = core.std.BlankClip(
        width=1920, height=1080, format=vs.RGBS, color=[0.2, 0.4, 0.6])
    for _ in range(5):
        for name, kernel_arguments in FUNCTIONS.items():
            large_output = getattr(core.dsmvc, name)(
                large_source, width=1692, height=952,
                src_left=0.2222222222221717, src_top=0.25,
                src_width=1691.5555555555557, src_height=951.5,
                backend="cpu", **kernel_arguments)
            large_output.get_frame(0)
            large_output.get_frame(0)

    test_cpu_dynamic_route(core, options.threads)

    print("dsmvc VapourSynth integration tests passed")


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    vs_root = Path(r"D:\okegui\OKEGui\tools\vapoursynth")
    result = argparse.ArgumentParser()
    result.add_argument("--plugin", default=str(root / "build" / "Release" / "dsmvc.dll"))
    result.add_argument("--old-plugin", default=str(
        vs_root / "vapoursynth64" / "plugins" / "descale.dll"))
    result.add_argument("--old-wrapper", default="")
    result.add_argument("--vs-root", default=str(vs_root))
    result.add_argument("--repo-root", default=str(root))
    result.add_argument("--threads", type=int, default=32)
    result.add_argument("--cuda-enabled", action="store_true")
    result.add_argument("--vulkan-enabled", action="store_true")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
