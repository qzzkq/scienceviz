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

## src/core

Ядро симулятора: библиотека `svcore`, от рендера не зависит. Пока в ней сетка
`svcore::Grid3D` — локальный блок `nx*ny*nz` своих ячеек плюс гало толщиной 1
или 2. На каждое поле (`rho`, `mom_x`, `mom_y`, `mom_z`, `phi`) свой плоский
массив `float`.

Индекс x самый быстрый, как в `Volume`, поэтому плоскость `z = const` лежит в
памяти одним куском, а слой гало по z — тоже: при декомпозиции слоями по z его
можно отдать в `MPI_Sendrecv` как есть (`z_send_lo`, `z_send_hi`, `z_recv_lo`,
`z_recv_hi`, `z_slab_count`). Свои ячейки это `[0, n)`, гало адресуется теми же
функциями индексами от `-halo` до `n + halo`; проверить попадание —
`contains_own` и `contains_with_halo`. Для одного процесса гало заполняется
периодически: `fill_halo_periodic`.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure   # scienceviz_tests и svcore_tests
ctest --test-dir build -R svcore --output-on-failure   # только ядро
```
