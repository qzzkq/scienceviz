#!/usr/bin/env python3
"""Record repeated local MPI diffusion timings; verification stays enabled."""
import argparse
import csv
import datetime
import json
import pathlib
import platform
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=pathlib.Path, default=pathlib.Path("build-mpi-release"))
    parser.add_argument("--mpiexec", default="mpiexec")
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("out/mpi-benchmark"))
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("use at least 3 repeats")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    executable = args.build.resolve() / "mpi_diffusion"
    rows = []
    # Каждый набор рангов прогревается; далее размеры чередуются по повтору.
    for repeat in range(-1, args.repeats):
        for ranks in (1, 2, 3):
            command = [args.mpiexec, "-n", str(ranks), str(executable), "--size", str(args.size),
                       "--steps", str(args.steps), "--halo", "1"]
            result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=120)
            report = json.loads(result.stdout)
            if not report["verified"]:
                raise RuntimeError("diffusion verification failed")
            if repeat >= 0:
                report["repeat"] = repeat + 1
                rows.append(report)
                (args.output_dir / f"r{ranks}-repeat{repeat+1}.log").write_text(
                    "COMMAND: " + repr(command) + "\nSTDOUT:\n" + result.stdout + "\nSTDERR:\n" + result.stderr)
    with (args.output_dir / "timings.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "timings-phases.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["processes", "phase", "elapsed_ms"])
        for row in rows:
            for phase, key in (("compute", "compute_s_max"), ("halo", "halo_s_max"), ("total", "loop_s_max")):
                writer.writerow([row["ranks"], phase, 1000 * row[key]])
    summary = {"recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "host": platform.node(), "scope": "one physical host; no affinity configured by script",
               "size": args.size, "steps": args.steps, "repeats": args.repeats, "ranks": []}
    for ranks in (1, 2, 3):
        selected = [row for row in rows if row["ranks"] == ranks]
        entry = {"ranks": ranks}
        for key in ("compute_s_max", "halo_s_max", "loop_s_max"):
            values = [row[key] for row in selected]
            entry[key] = {"median": statistics.median(values), "min": min(values), "max": max(values)}
        summary["ranks"].append(entry)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
