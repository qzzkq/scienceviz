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

## От Python-расчёта к кадру

Для расчёта нужны Python, NumPy версии 2 или выше, SciPy и Matplotlib. Сначала соберите C++-код командами выше. Затем из корня репозитория:

```bash
python3 prototype/render_snapshot.py --size 16 --out prototype/output/frame.png
```

Команда считает адиабатический газ в Python, передаёт плотность в C++-сетку с периодическим гало и рендерит её. Вместо нового расчёта можно загрузить ранее сохранённый архив прототипа:

```bash
python3 prototype/render_snapshot.py --snapshot prototype/output/web3d_64.npz --out prototype/output/frame.png
```

`--matter` выбирает `adiabatic`, `isothermal`, `dust` или `dm`; `--a` — момент снимка: `0.1`, `0.25`, `0.5` или `1.0`. `--dump prototype/output/cube.f32` дополнительно сохраняет куб для рендера. В Python плотность индексируется как `[x,y,z]`, при передаче в C++ оси переставляются так, чтобы x был самым быстрым индексом. Для передачи используется float32.

Расчёт по-прежнему выполняется в Python: C++-сетка пока не реализует эволюцию газа, гравитацию или MPI. Передаточная функция рендера пока настроена на синтетические данные.

`ctest --test-dir build --output-on-failure` проверяет также передачу снимка через сетку и рендер, если при конфигурации CMake найден Python с NumPy.

## Пролёт камеры и переход между снимками

Для видео нужны Python с NumPy и FFmpeg с кодировщиком `libx264`. Из корня репозитория:

```bash
python prototype/render_video.py --npy data/camels_mgas_0.npy --camera examples/orbit.json --out build/camels.mp4
python prototype/render_video.py --snapshot prototype/output/web3d_64.npz --camera examples/orbit.json --out build/evolution.mp4
```

В JSON-файле `eyes` задаёт точки сплайна Катмулла–Рома, `target` — общую точку взгляда.
Координаты куба идут от 0 до 1. Для одного PNG можно передать `--eye x,y,z`
и `--target x,y,z` напрямую в `scienceviz`.
Для архива плотности линейно смешиваются между соседними снимками с равномерным шагом по масштабному фактору.
`--matter` выбирает поле, `--frames` и `--fps` — число кадров и частоту.
Имя программы можно задать через `--renderer`, путь к FFmpeg — через `--ffmpeg`.

Для финального видео используйте `examples/final-flythrough.json`. Камера
показывает куб общим планом, плавно опускается и приближается, затем отходит.
При 240 кадрах и 24 fps пролёт занимает 10 секунд.

```bash
python prototype/render_video.py --snapshot prototype/output/web3d_64.npz --camera examples/final-flythrough.json --frames 240 --fps 24 --out build/final-flythrough.mp4
```

Если CMake находит OpenMP, рендер параллелит строки кадра. Число потоков можно
задать переменной окружения `OMP_NUM_THREADS`. Для контрольного замера используйте
одинаковый куб, камеру и разрешение с одним и несколькими потоками.

## Графики замеров MPI

```bash
python prototype/plot_timings.py build/mpi-timings.csv --out build/mpi-timings.png
```

Скрипту нужен Matplotlib и CSV с колонками `processes,phase,elapsed_ms`.
Одна строка — время одной фазы в миллисекундах. Для повторов используется медиана.
Фаза `total` содержит измеренное полное время, включая запуск на одном процессе.
Ускорение считается как `T(1) / T(p)` по `total`. Времена отдельных фаз
могут относиться к разным рангам MPI, поэтому на графике они показаны отдельными линиями.

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
