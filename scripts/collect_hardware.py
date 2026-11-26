#!/usr/bin/env python3
"""Collect non-secret hardware facts for the MPI node inventory (no IPs or keys)."""
import argparse
import datetime
import json
import os
import pathlib
import platform
import subprocess


def output(command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL,
                                       env={**os.environ, "LC_ALL": "C"}, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", required=True, help="Team node label; no username or IP needed")
    args = parser.parse_args()
    cpu = {}
    cpu_raw = output(["lscpu", "--json"])
    if cpu_raw != "unavailable":
        cpu = {item["field"].rstrip(":"): item["data"] for item in json.loads(cpu_raw)["lscpu"]}
    memory = pathlib.Path("/proc/meminfo").read_text()
    mem_kib = int(next(line.split()[1] for line in memory.splitlines() if line.startswith("MemTotal:")))
    pci = output(["lspci"])
    result = {
        "node": args.node,
        "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "os": platform.freedesktop_os_release().get("PRETTY_NAME", platform.system()),
        "architecture": platform.machine(),
        "cpu_model": cpu.get("Model name", "unavailable"),
        "logical_cpus": os.cpu_count(),
        "memory_gib": round(mem_kib / 1048576, 2),
        "gpu": [line for line in pci.splitlines() if any(s in line.lower() for s in ("vga", "3d controller", "display controller"))],
        "network_adapters": [line for line in pci.splitlines() if any(s in line.lower() for s in ("ethernet", "network controller"))],
        "mpi_launcher": output(["mpiexec", "--version"]),
        "compiler": output(["c++", "--version"]).splitlines()[0],
        "cmake": output(["cmake", "--version"]).splitlines()[0],
        "wired_link_and_ssh": "must be checked separately",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
