# scienceviz

Офлайн-рендер куба плотности. Синтетическое поле, проход лучей, кадр PNG.

```bash
cmake -S . -B build
cmake --build build
./build/scienceviz --size 64 --out frame.png
ctest --test-dir build --output-on-failure
```

На Windows исполняемый файл — `build/scienceviz.exe`.

Куб пишется и читается как `cube.f32` (`--dump` / `--bin`). Один эталонный куб float32 — `--npy`. Откуда брать CAMELS и Quijote: [data/README.md](data/README.md).
