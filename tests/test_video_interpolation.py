"""Проверка ключевых точек камеры и интерполяции плотности между снимками."""

from pathlib import Path
import sys
import tempfile

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "prototype"))
from render_snapshot import render_order
from render_video import camera_path, camera_position, snapshot_frames

eyes, _ = camera_path(root / "examples" / "orbit.json")
np.testing.assert_allclose(
    [camera_position(eyes, i / (len(eyes) - 1)) for i in range(len(eyes))], eyes
)
step = 1e-5
for i in range(1, len(eyes) - 1):
    p = i / (len(eyes) - 1)
    left = (camera_position(eyes, p) - camera_position(eyes, p - step)) / step
    right = (camera_position(eyes, p + step) - camera_position(eyes, p)) / step
    np.testing.assert_allclose(left, right, atol=1e-3)

with tempfile.TemporaryDirectory(prefix="scienceviz-test-") as directory:
    archive = Path(directory) / "snapshots.npz"
    start = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    np.savez(archive, **{
        "adiabatic_a1.000": start + 10,
        "adiabatic_a0.200": start,
        "adiabatic_a0.400": start + 4,
    })
    frames = list(snapshot_frames(archive, "adiabatic", 5))
    assert len(frames) == 5
    for frame, offset in zip(frames, (0, 4, 6, 8, 10)):
        np.testing.assert_allclose(frame, render_order(start + offset))

    np.savez(archive, **{"adiabatic_a0.200": start, "adiabatic_a1.000": np.ones((3, 3, 3))})
    try:
        list(snapshot_frames(archive, "adiabatic", 2))
    except ValueError as error:
        assert "размеры сеток" in str(error)
    else:
        raise AssertionError("снимки с разными размерами приняты")
