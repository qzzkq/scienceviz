#!/usr/bin/env python3
"""Run real MPI executables, retain evidence, compare complete 1/2/3-rank fields."""
import argparse
import array
import hashlib
import json
import pathlib
import platform
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=pathlib.Path, default=pathlib.Path("build-mpi"))
    parser.add_argument("--mpiexec", default="mpiexec")
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("out/mpi-validation"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {"host": platform.node(), "scope": "local MPI; not a multi-machine network benchmark", "runs": [], "comparisons": []}

    def run(ranks, executable, parameters, name):
        command = [args.mpiexec, "-n", str(ranks), str(args.build.resolve() / executable), *parameters]
        process = subprocess.run(command, text=True, capture_output=True, timeout=120)
        (args.output_dir / (name + ".log")).write_text(
            "COMMAND: " + repr(command) + "\nSTDOUT:\n" + process.stdout + "\nSTDERR:\n" + process.stderr)
        if process.returncode:
            raise RuntimeError(f"{name}: exit {process.returncode}; see log")
        return process.stdout

    for ranks in (1, 2, 3):
        run(ranks, "mpi_hello", [], f"hello-{ranks}")
    for nx, ny, nz, halo, steps in ((17, 13, 19, 1, 20), (17, 13, 19, 2, 20), (8, 6, 6, 2, 7)):
        fields = []
        name = f"{nx}x{ny}x{nz}-g{halo}"
        for ranks in (1, 2, 3):
            dump = args.output_dir.resolve() / f"{name}-r{ranks}.f32"
            stdout = run(ranks, "mpi_diffusion", ["--nx", str(nx), "--ny", str(ny), "--nz", str(nz),
                "--halo", str(halo), "--steps", str(steps), "--output", str(dump)], f"{name}-r{ranks}")
            report = json.loads(stdout)
            if not report["verified"]:
                raise RuntimeError(f"{name}: numerical verification failed")
            payload = dump.read_bytes()
            if len(payload) != nx * ny * nz * 4:
                raise RuntimeError(f"{name}: wrong field length")
            report["field_sha256"] = hashlib.sha256(payload).hexdigest()
            results["runs"].append(report)
            field = array.array("f")
            field.frombytes(payload)
            fields.append((payload, field))
        for index, ranks in enumerate((2, 3), 1):
            error = max(abs(a-b) for a,b in zip(fields[0][1], fields[index][1]))
            if error > 1e-6:
                raise RuntimeError(f"{name}: 1 vs {ranks} ranks differs by {error}")
            results["comparisons"].append({"case": name, "ranks": [1, ranks],
                "max_abs_error": error, "bitwise_equal": fields[0][0] == fields[index][0]})
    run(2, "mpi_pingpong", [], "pingpong-local")
    (args.output_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "runs": len(results["runs"]),
                      "comparisons": results["comparisons"]}, indent=2))


if __name__ == "__main__":
    main()
