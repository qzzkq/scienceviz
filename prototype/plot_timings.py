"""Графики времени фаз и ускорения по замерам на нескольких процессах."""

import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_timings(path):
    samples = defaultdict(lambda: defaultdict(list))
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        columns = {"processes", "phase", "elapsed_ms"}
        if not columns.issubset(rows.fieldnames or ()):
            raise ValueError("в CSV нужны колонки processes, phase, elapsed_ms")
        for row in rows:
            try:
                processes = int(row["processes"])
                elapsed = float(row["elapsed_ms"])
                phase = (row["phase"] or "").strip()
                if processes < 1 or not phase or not math.isfinite(elapsed) or elapsed < 0:
                    raise ValueError
                if phase == "total" and elapsed == 0:
                    raise ValueError
            except (TypeError, ValueError) as exc:
                raise ValueError(f"некорректный замер в строке {rows.line_num}") from exc
            samples[phase][processes].append(elapsed)
    if 1 not in samples["total"]:
        raise ValueError("для ускорения нужен замер фазы total на одном процессе")
    return {
        phase: {processes: median(values) for processes, values in sorted(by_count.items())}
        for phase, by_count in samples.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--title", default="Замеры MPI")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    try:
        timings = read_timings(args.csv)
        counts = sorted({count for series in timings.values() for count in series})
        fig, (time_axis, speed_axis) = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
        fig.suptitle(args.title)

        for phase, values in timings.items():
            time_axis.plot(list(values), list(values.values()), marker="o", label=phase)
        time_axis.set(title="Время фаз", ylabel="Медиана, мс", xticks=counts)
        time_axis.legend()

        totals = timings["total"]
        speed_axis.plot(list(totals), [totals[1] / elapsed for elapsed in totals.values()],
                        marker="o", label="по фазе total")
        speed_axis.plot(counts, counts, linestyle="--", color="#777777", label="идеальное")
        speed_axis.set(title="Ускорение", ylabel="T(1) / T(p)", xticks=counts)
        speed_axis.legend()

        for axis in (time_axis, speed_axis):
            axis.set_xlabel("Число процессов")
            axis.grid(alpha=0.2)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=160)
        plt.close(fig)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(1, f"ошибка: {exc}\n")


if __name__ == "__main__":
    main()
