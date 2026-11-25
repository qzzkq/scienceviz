"""Compare the C++ shock tube with the Python reference and verify restart."""
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "prototype"))
from hydro1d import Gas, Sim1D

solver = Path(sys.argv[1]).resolve()


def run(path, settings, succeeds=True):
    path.write_text(settings)
    result = subprocess.run([str(solver), str(path)], capture_output=True, text=True)
    assert (result.returncode == 0) == succeeds, result.stderr
    return result


with tempfile.TemporaryDirectory(prefix="scienceviz-sim1d-") as directory:
    tmp = Path(directory)
    n = 100
    prefix = tmp / "sod"
    base = f"""cells={n}
length=1
cfl=0.4
order=2
gas=adiabatic
gamma=1.4
boundary=outflow
initial=sod
output_prefix={prefix}
"""
    run(tmp / "sod.cfg", base + "end_time=0.05\n")
    result = np.genfromtxt(str(prefix) + ".csv", delimiter=",", names=True,
                           comments="#", skip_header=1)
    rho = np.where(np.arange(n) < n // 2, 1.0, 0.125)
    pressure = np.where(np.arange(n) < n // 2, 1.0, 0.1)
    gas = Gas("adiabatic", gamma=1.4, dual_energy=False)
    reference = Sim1D(gas, gas.conserved(rho, np.zeros(n), pressure, 1.0),
                      1.0, order=2, bc="outflow")
    reference.run_to_time(0.05)
    for field, expected in (("rho", reference.U[0]), ("mom", reference.U[1]),
                            ("energy", reference.U[2])):
        assert np.max(np.abs(result[field] - expected)) < 2e-10, field
    assert abs(result["rho"].sum() / n - reference.mass()) < 1e-12

    checkpoint = str(prefix) + ".chk"
    run(tmp / "resume.cfg", base + f"end_time=0.1\nrestart={checkpoint}\n")
    resumed = np.genfromtxt(str(prefix) + ".csv", delimiter=",", names=True,
                            comments="#", skip_header=1)
    reference.run_to_time(0.1)
    assert np.max(np.abs(resumed["rho"] - reference.U[0])) < 2e-10

    run(tmp / "bad.cfg", base + "end_time=nan\n", succeeds=False)
    run(tmp / "bad.cfg", base + "end_time=0.1\ncells=200\n", succeeds=False)
    run(tmp / "bad.cfg", base + "end_time=0.1\nrestart=missing.chk\n", succeeds=False)
