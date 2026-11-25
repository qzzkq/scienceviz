"""Пролёт камеры над одним кубом или последовательностью снимков."""

import argparse
from bisect import bisect_right
from functools import lru_cache
from itertools import repeat
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from zipfile import BadZipFile

import numpy as np

from render_snapshot import render_order


def camera_path(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("траектория камеры должна быть объектом JSON")
    eyes = np.asarray(config.get("eyes"), dtype=float)
    target = np.asarray(config.get("target", [0.5, 0.5, 0.5]), dtype=float)
    if eyes.ndim != 2 or eyes.shape[0] < 2 or eyes.shape[1] != 3 or not np.isfinite(eyes).all():
        raise ValueError("в eyes нужны хотя бы две точки [x, y, z]")
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("в target нужна точка [x, y, z]")
    return eyes, target


def camera_position(eyes, progress):
    last = len(eyes) - 1
    position = progress * last
    segment = min(int(position), last - 1)
    t = position - segment
    p0 = eyes[max(segment - 1, 0)]
    p1, p2 = eyes[segment:segment + 2]
    p3 = eyes[min(segment + 2, last)]
    # Catmull–Rom, с повтором крайних точек на концах траектории.
    return 0.5 * (
        2 * p1 + (p2 - p0) * t
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2
        + (3 * p1 - p0 - 3 * p2 + p3) * t**3
    )


def snapshot_frames(path, matter, count):
    prefix = f"{matter}_a"
    archive = np.load(path, allow_pickle=False)
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError("--snapshot требует архив .npz")
    with archive:
        snapshots = sorted((float(key[len(prefix):]), key) for key in archive.files if key.startswith(prefix))
        if not snapshots:
            raise ValueError(f"в архиве нет снимков {matter}")
        times, keys = zip(*snapshots)
        if not all(math.isfinite(time) and time > 0 for time in times) or len(set(times)) != len(times):
            raise ValueError("масштабные факторы в архиве должны быть положительными, конечными и уникальными")

        shape = None

        @lru_cache(maxsize=2)
        def load(index):
            nonlocal shape
            density = archive[keys[index]]
            if shape is None:
                shape = density.shape
            if density.shape != shape:
                raise ValueError("размеры сеток снимков должны совпадать")
            return render_order(density)

        for time in np.linspace(times[0], times[-1], count):
            left = bisect_right(times, time) - 1
            before = load(left)
            if left == len(times) - 1:
                yield before
                continue
            after = load(left + 1)
            fraction = np.float32((time - times[left]) / (times[left + 1] - times[left]))
            yield (np.float32(1) - fraction) * before + fraction * after


def point_arg(point):
    return ",".join(f"{coordinate:.9g}" for coordinate in point)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--npy", type=Path, help="один куб float32 в порядке рендера")
    source.add_argument("--snapshot", type=Path, help="архив web3d_*.npz со снимками")
    parser.add_argument("--matter", choices=("adiabatic", "isothermal", "dust", "dm"), default="adiabatic")
    parser.add_argument("--camera", required=True, type=Path, help="JSON с ключевыми точками камеры")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--out", required=True, type=Path)
    default_renderer = "scienceviz.exe" if os.name == "nt" else "scienceviz"
    parser.add_argument("--renderer", type=Path, default=Path(__file__).resolve().parents[1] / "build" / default_renderer)
    parser.add_argument("--ffmpeg", default="ffmpeg", help="команда или путь к FFmpeg")
    args = parser.parse_args()
    if args.frames < 2 or args.fps < 1 or args.out.suffix.lower() != ".mp4":
        parser.error("нужны хотя бы два кадра, положительный fps и выходной файл .mp4")
    if args.snapshot and args.snapshot.suffix.lower() != ".npz":
        parser.error("--snapshot требует архив .npz")
    ffmpeg = shutil.which(args.ffmpeg)
    if not ffmpeg:
        parser.error("FFmpeg не найден. Добавьте его в PATH или укажите --ffmpeg")
    if not args.renderer.is_file():
        parser.error(f"рендер не найден: {args.renderer}. Сначала соберите проект через CMake")

    try:
        eyes, target = camera_path(args.camera)
        output = args.out.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="scienceviz-video-", dir=output.parent) as directory:
            work = Path(directory)
            cube = work / "density.npy" if args.snapshot else args.npy.resolve()
            densities = repeat(None, args.frames)
            if args.snapshot:
                densities = snapshot_frames(args.snapshot, args.matter, args.frames)
            for index, density in enumerate(densities):
                if density is not None:
                    np.save(cube, density, allow_pickle=False)
                eye = camera_position(eyes, index / (args.frames - 1))
                subprocess.run([
                    str(args.renderer.resolve()), "--npy", str(cube),
                    "--eye", point_arg(eye), "--target", point_arg(target),
                    "--out", str(work / f"frame-{index:06d}.png"),
                ], check=True)
            video = work / "video.mp4"
            subprocess.run([
                ffmpeg, "-hide_banner", "-loglevel", "error", "-framerate", str(args.fps),
                "-i", str(work / "frame-%06d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(video),
            ], check=True)
            video.replace(output)
        print(output)
    except (BadZipFile, OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"ошибка: {exc}\n")


if __name__ == "__main__":
    main()
