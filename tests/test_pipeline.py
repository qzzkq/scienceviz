"""Проверка пути NPZ → C++-сетка → изображение и куб рендера."""
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

root = Path(__file__).resolve().parent.parent
renderer = Path(sys.argv[1]).resolve()
script = root / "prototype" / "render_snapshot.py"

with tempfile.TemporaryDirectory(prefix="scienceviz-test-") as tmp:
    tmp = Path(tmp)
    x, y, z = np.indices((4, 4, 4))
    density = (1 + x + 10 * y + 100 * z).astype(np.float32)
    archive = tmp / "snapshots.npz"
    np.savez_compressed(archive, **{"adiabatic_a1.000": density})
    image = tmp / "frame.png"
    cube = tmp / "cube.f32"
    command = [sys.executable, str(script), "--snapshot", str(archive), "--renderer", str(renderer),
               "--out", str(image), "--dump", str(cube)]
    subprocess.run(command, check=True)
    assert image.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    with cube.open("rb") as stream:
        assert stream.read(8) == b"SVZCUBE1"
        assert int.from_bytes(stream.read(4), "little") == 4
        actual = np.frombuffer(stream.read(), dtype="<f4").reshape((4, 4, 4))
    assert np.array_equal(actual, density.transpose(2, 1, 0))
    missing = subprocess.run(command + ["--a", "0.5"], capture_output=True, text=True)
    assert missing.returncode != 0 and "в архиве нет снимка" in missing.stderr
