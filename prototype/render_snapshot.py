"""Расчёт или загрузка снимка Python → C++-сетка → изображение рендера."""
import argparse
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def render_order(density):
    """Python хранит [x,y,z], C++ читает файл с самым быстрым индексом x."""
    density = np.asarray(density)
    if density.ndim != 3 or len(set(density.shape)) != 1 or not 1 <= density.shape[0] <= 512:
        raise ValueError("нужен куб плотности N×N×N, где 1 ≤ N ≤ 512")
    if not np.isfinite(density).all() or (density < 0).any():
        raise ValueError("плотность должна быть конечной и неотрицательной")
    if density.max() > np.finfo(np.float32).max:
        raise ValueError("плотность не помещается в float32")
    return np.ascontiguousarray(density.transpose(2, 1, 0), dtype="<f4")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, help="готовый архив web3d_*.npz; без него выполняется расчёт")
    parser.add_argument("--size", type=int, default=16, help="сторона новой сетки (по умолчанию 16)")
    parser.add_argument("--length", type=float, default=64.0, help="сторона куба, Мпк/h")
    parser.add_argument("--matter", choices=("adiabatic", "isothermal", "dust", "dm"), default="adiabatic")
    parser.add_argument("--a", type=float, choices=(0.1, 0.25, 0.5, 1.0), default=1.0)
    parser.add_argument("--renderer", type=Path, default=Path(__file__).resolve().parent.parent / "build" / "scienceviz")
    parser.add_argument("--out", type=Path, default=Path("frame.png"))
    parser.add_argument("--dump", type=Path, help="дополнительно сохранить куб в формате рендера")
    args = parser.parse_args()

    key = f"{args.matter}_a{args.a:.3f}"
    try:
        if not args.renderer.is_file():
            raise ValueError(f"рендер не найден: {args.renderer}. Сначала собери проект через cmake")
        if args.snapshot:
            with np.load(args.snapshot, allow_pickle=False) as archive:
                if key not in archive:
                    raise ValueError(f"в архиве нет снимка {key}")
                density = archive[key]
        else:
            if args.size < 4 or args.size > 512 or args.size % 2 or not np.isfinite(args.length) or args.length <= 0:
                raise ValueError("для расчёта нужны чётный размер от 4 до 512 и положительная сторона куба")
            from cosmology import Cosmology
            from web3d import initial_fields, run_dark_matter, run_gas

            cosmo = Cosmology()
            delta_k, delta, velocity = initial_fields(args.size, args.length, cosmo)
            snapshots = {}
            if args.matter == "dm":
                run_dark_matter(delta_k, args.size, args.length, cosmo, snapshots)
            else:
                run_gas(args.matter, delta, velocity, args.size, args.length, cosmo, snapshots)
            density = snapshots[key]

        args.out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="scienceviz-") as tmp:
            path = Path(tmp) / "density.npy"
            np.save(path, render_order(density), allow_pickle=False)
            command = [str(args.renderer.resolve()), "--grid-npy", str(path), "--out", str(args.out)]
            if args.dump:
                command.extend(["--dump", str(args.dump)])
            subprocess.run(command, check=True)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"ошибка: {exc}\n")


if __name__ == "__main__":
    main()
