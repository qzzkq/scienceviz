# Эталонные кубы

Рендер читает один куб float32 в порядке C, форма `(N, N, N)`. Стек из многих кубов в одном файле он не берёт.

## CAMELS Multifield Dataset

Плотность газа и тёмной материи на сетках 128³, 256³ и 512³. Описание полей: [data.html](https://camels-multifield-dataset.readthedocs.io/en/latest/data.html). Как скачать (Globus или URL): [access.html](https://camels-multifield-dataset.readthedocs.io/en/latest/access.html).

Имена 3D-файлов: `Grids_<поле>_<набор>_<сэмпл>_<N>_z=<z>.npy`.

- плотность газа: `Grids_Mgas_IllustrisTNG_LH_128_z=0.0.npy`
- плотность тёмной материи: `Grids_Mcdm_IllustrisTNG_LH_128_z=0.0.npy`

Один воксель — float32. Один куб 128³ занимает 8 МБ, 256³ — 64 МБ. В файле LH лежит 1000 таких кубов, поэтому файл 128³ — около 7.8 ГБ, 256³ — около 62.5 ГБ, 512³ — около 500 ГБ. Целиком качать не нужно.

Для первого прогона подходит набор EX из четырёх кубов 128³. Файл
`Grids_Mgas_IllustrisTNG_EX_128_z=0.0.npy` весит 32 МБ и лежит в каталоге
`3D_grids/data/IllustrisTNG/`. [Скачать набор EX](https://users.flatironinstitute.org/~fvillaescusa/priv/DEPnzxoWlaTQ6CjrXqsm0vYi8L7Jy/CMD/3D_grids/data/IllustrisTNG/Grids_Mgas_IllustrisTNG_EX_128_z=0.0.npy).

Из корня репозитория сохраните один куб в порядке рендера `[z,y,x]`. Деление
на среднюю плотность переводит поле в шкалу передаточной функции:

```python
import numpy as np

grids = np.load("Grids_Mgas_IllustrisTNG_EX_128_z=0.0.npy", mmap_mode="r")
density = grids[0]
normalized = density.transpose(2, 1, 0) / density.mean(dtype=np.float64)
np.save("data/camels_mgas_0.npy", np.ascontiguousarray(normalized, dtype="<f4"))
```

`scienceviz --npy data/camels_mgas_0.npy --out frame.png` создаст кадр.

## Quijote

Кубы плотности: [Density fields](https://quijote-simulations.readthedocs.io/en/latest/df.html). Они лежат на кластере авторов в `3D_cubes`, файл уже один куб: `df_m_128_CIC_z=0.npy` (128³, около 8 МБ), также 256 и 512, схемы CIC и PCS. Крупные сетки на кластере могут отсутствовать. Публичной маленькой HTTP-ссылки на весь набор нет: смотреть раздел Data access в документации Quijote.
