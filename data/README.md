# Эталонные кубы

Рендер читает один куб float32 в порядке C, форма `(N, N, N)`. Стек из многих кубов в одном файле он не берёт.

## CAMELS Multifield Dataset

Плотность газа и тёмной материи на сетках 128³, 256³ и 512³. Описание полей: [data.html](https://camels-multifield-dataset.readthedocs.io/en/latest/data.html). Как скачать (Globus или URL): [access.html](https://camels-multifield-dataset.readthedocs.io/en/latest/access.html).

Имена 3D-файлов: `Grids_<поле>_<набор>_<сэмпл>_<N>_z=<z>.npy`.

- плотность газа: `Grids_Mgas_IllustrisTNG_LH_128_z=0.00.npy`
- плотность тёмной материи: `Grids_Mcdm_IllustrisTNG_LH_128_z=0.00.npy`

Один воксель — float32. Один куб 128³ занимает 8 МБ, 256³ — 64 МБ. В файле LH лежит 1000 таких кубов, поэтому файл 128³ — около 7.8 ГБ, 256³ — около 62.5 ГБ, 512³ — около 500 ГБ. Целиком качать не нужно.

Форма файла — `(1000, N, N, N)`. Для рендера вытащить один куб:

```python
import numpy as np

grids = np.load("Grids_Mgas_IllustrisTNG_LH_128_z=0.00.npy", mmap_mode="r")
np.save("camels_mgas_0.npy", np.asarray(grids[0]))
```

Дальше: `scienceviz --npy camels_mgas_0.npy --out frame.png`.

## Quijote

Кубы плотности: [Density fields](https://quijote-simulations.readthedocs.io/en/latest/df.html). Они лежат на кластере авторов в `3D_cubes`, файл уже один куб: `df_m_128_CIC_z=0.npy` (128³, около 8 МБ), также 256 и 512, схемы CIC и PCS. Крупные сетки на кластере могут отсутствовать. Публичной маленькой HTTP-ссылки на весь набор нет: смотреть раздел Data access в документации Quijote.
