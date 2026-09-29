# Научная литература для проекта: распределённый симулятор космической паутины

Статьи сгруппированы по частям проекта: зачем каждая нужна и где её найти.

## 0. Начать с этого
- **Angulo & Hahn (2022), «Large-scale dark matter simulations»**, Living Reviews in Computational Astrophysics. [arXiv:2112.05165](https://arxiv.org/abs/2112.05165). Обзор, покрывающий почти весь проект: уравнения в сопутствующих координатах, сравнение N-body/PM и альтернативных методов (Эйлер, Власов, «листы» из тетраэдров, Шрёдингер), интегрирование по времени, начальные условия. Хорошо подходит как основная ссылка в отчёте.

## 1. Физика: Эйлер или PM и проблема пересечения потоков (Человек 1)
- **Trac & Pen (2003), «A Primer on Eulerian CFD for Astrophysics»**, PASP 115, 303. [astro-ph/0210611](https://arxiv.org/abs/astro-ph/0210611). Учебное введение в эйлерову сеточную гидродинамику: потоки через грани, CFL, схемы. Авторы также делают PM-коды.
- **Gurbatov, Saichev, Shandarin (2012), «Крупномасштабная структура Вселенной. Приближение Зельдовича и модель слипания»**, УФН 182, 233 ([DOI](https://iopscience.iop.org/article/10.3367/UFNe.0182.201203a.0233/meta)). Статья на русском. Модель слипания (уравнение Бюргерса) даёт эйлеровому «газу без давления» способ пережить пересечение потоков за счёт малой вязкости. Это прямой ответ на главный риск вашей схемы.
- **Hahn, Abel, Kaehler (2013), «A new approach to simulating collisionless dark matter fluids»**, MNRAS 434, 1171. [arXiv:1210.6652](https://arxiv.org/abs/1210.6652). Описывает холодную тёмную материю как 3D-многообразие в фазовом пространстве: точная плотность и многопотоковость.
- **Yoshikawa, Yoshida, Umemura (2013)**: прямое решение уравнений Власова–Пуассона в 6D-фазовом пространстве, ApJ 762, 116. [arXiv:1206.6152](https://arxiv.org/abs/1206.6152). Это «честная» сеточная альтернатива частицам, а в работе на Fugaku ([SC'21](https://dl.acm.org/doi/abs/10.1145/3458817.3487401)) она масштабирована до 400 трлн ячеек.
- **Mocz et al. (2018), «On the Schrödinger-Poisson–Vlasov-Poisson correspondence»**. [arXiv:1801.03507](https://arxiv.org/abs/1801.03507). Волновой подход тоже хранит только 3D-сетку, но воспроизводит многопотоковость. Близкая работа: Uhlemann et al. (2019), [arXiv:1812.05633](https://arxiv.org/abs/1812.05633).
- **Libeskind et al. (2018), «Tracing the cosmic web»**, MNRAS 473, 1195. [arXiv:1705.03021](https://arxiv.org/abs/1705.03021). Сравнивает методы, которые выделяют в симуляции волокна, узлы и войды. Пригодится для раздела валидации.

## 2. Коды-образцы: сетка, PM, GPU и MPI (Человек 2–3)
- **Almgren et al. (2013), Nyx**, ApJ 765, 39 ([ADS](https://ui.adsabs.harvard.edu/abs/2013ApJ...765...39A)). Сочетает оба подхода: газ считается эйлерово на сетке, тёмная материя — частицами методом PM. Это прямое подтверждение рекомендации «PM для тёмной материи».
- **Schneider & Robertson (2015), Cholla**, ApJS 217, 24. [arXiv:1410.4194](https://arxiv.org/abs/1410.4194). Эйлеров код на GPU с MPI-декомпозицией на схеме «один процесс — один GPU». Подробно разобраны ghost-ячейки и отношение поверхности к объёму. Это готовый образец для вашего обмена гало.
- **Feng, Chu, Seljak, McDonald (2016), FastPM**, MNRAS. [arXiv:1603.00476](https://arxiv.org/abs/1603.00476). Простой масштабируемый PM с 2D-декомпозицией. Хороший образец, если выберете PM.
- **Yu, Pen et al. (2018), CUBE**, ApJS. [arXiv:1712.06121](https://arxiv.org/abs/1712.06121). Хранит данные частиц в 1-байтовом формате с фиксированной точкой, около 6 байт на частицу. Это прямо про ваше квантование и экономию памяти. Продолжение — CUBE2, [arXiv:2512.12629](https://arxiv.org/abs/2512.12629).
- Код ориентированного на GPU PM-солвера HACC описан в Habib et al. (2016), [arXiv:1410.2805](https://arxiv.org/abs/1410.2805). RAMSES: Teyssier (2002), [astro-ph/0111367](https://arxiv.org/abs/astro-ph/0111367).

## 3. Пуассон через БПФ и распределённое БПФ (Человек 3)
- **Pekurovsky (2012), P3DFFT**, SIAM J. Sci. Comput. и **Li & Laizet (2010), 2DECOMP&FFT** (CUG 2010, [Semantic Scholar](https://www.semanticscholar.org/paper/2DECOMP&FFT-A-Highly-Scalable-2D-Decomposition-and-Li-Laizet/963b75d2215dfae94ccc3e02f629876f7608f79e)). Декомпозиция на «карандаши» (pencil) против «слоёв» (slab) — основа для схемы разбиения на узлы.
- **AccFFT (Gholami et al. 2015)**: распределённое БПФ на CPU и GPU. [arXiv:1506.07933](https://arxiv.org/abs/1506.07933).
- Классика: Hockney & Eastwood, *Computer Simulation Using Particles* (книга), и Efstathiou et al. (1985), ApJS 57, 241 — метод PM, CIC-интерполяция, функции Грина.

## 4. Начальные условия
- **Michaux, Hahn, Rampf, Angulo (2021), monofonIC**: начальные условия в 3LPT, MNRAS 500, 663. [arXiv:2008.09588](https://arxiv.org/abs/2008.09588). Для курсового проекта хватит также MUSIC ([arXiv:1103.6031](https://arxiv.org/abs/1103.6031)) или 2LPTic ([astro-ph/0606505](https://arxiv.org/abs/astro-ph/0606505)).
- Исходная работа по приближению Зельдовича: Zel'dovich (1970), A&A 5, 84.

## 5. Сжатие данных
- **Tao et al. (2017)**: оптимизация сжатия SZ на данных HACC. [arXiv:1707.08205](https://arxiv.org/abs/1707.08205).
- **Jin et al. (2020)**: сжатие с потерями на GPU (SZ, ZFP) на космологических данных. [arXiv:2004.00224](https://arxiv.org/abs/2004.00224). Бенчмарк SDRBench содержит готовые поля плотности Nyx для тестов: [arXiv:2101.03201](https://arxiv.org/abs/2101.03201).

## 6. Визуализация (Человек 4)
- **Kaehler, Hahn, Abel (2012), «A Novel Approach to Visualizing Dark Matter Simulations»**, IEEE TVCG 18(12) ([PubMed](https://pubmed.ncbi.nlm.nih.gov/26357114/)). Объёмный рендер именно космической паутины.
- Для основ raymarching: Levoy (1988), «Display of surfaces from volume data», и Max (1995), «Optical models for direct volume rendering» — классика про модель emission–absorption.

## 7. Валидация и эталоны
- **Nelson et al. (2019), IllustrisTNG Public Data Release**. [arXiv:1812.05609](https://arxiv.org/abs/1812.05609). Открытые снимки и описание формата.
- **Schneider et al. (2016), «Matter power spectrum and the challenge of percent accuracy»**, JCAP. [arXiv:1503.05920](https://arxiv.org/abs/1503.05920). Как сравнивать спектры мощности между кодами (RAMSES, PKDGRAV3, GADGET-3).
- Millennium: Springel et al. (2005), Nature 435, 629, [astro-ph/0504097](https://arxiv.org/abs/astro-ph/0504097).


## Источники
- [arXiv:2112.05165](https://arxiv.org/abs/2112.05165), [Springer LRCA](https://link.springer.com/article/10.1007/s41115-021-00013-z)
- [arXiv:astro-ph/0210611](https://arxiv.org/abs/astro-ph/0210611)
- [УФН / Phys. Usp. 2012](https://iopscience.iop.org/article/10.3367/UFNe.0182.201203a.0233/meta)
- [arXiv:1210.6652](https://arxiv.org/abs/1210.6652)
- [arXiv:1206.6152](https://arxiv.org/abs/1206.6152), [Fugaku Vlasov SC'21](https://dl.acm.org/doi/abs/10.1145/3458817.3487401)
- [arXiv:1801.03507](https://arxiv.org/abs/1801.03507)
- [arXiv:1705.03021](https://arxiv.org/abs/1705.03021)
- [Nyx ADS](https://ui.adsabs.harvard.edu/abs/2013ApJ...765...39A)
- [arXiv:1410.4194](https://arxiv.org/abs/1410.4194)
- [arXiv:1603.00476](https://arxiv.org/abs/1603.00476)
- [arXiv:1712.06121](https://arxiv.org/pdf/1712.06121), [arXiv:2512.12629](https://arxiv.org/abs/2512.12629)
- [2DECOMP&FFT](https://www.semanticscholar.org/paper/2DECOMP&FFT-A-Highly-Scalable-2D-Decomposition-and-Li-Laizet/963b75d2215dfae94ccc3e02f629876f7608f79e), [AccFFT](https://arxiv.org/pdf/1506.07933)
- [arXiv:2008.09588](https://arxiv.org/pdf/2008.09588)
- [arXiv:1707.08205](https://arxiv.org/abs/1707.08205), [arXiv:2004.00224](https://arxiv.org/pdf/2004.00224), [SDRBench](https://ar5iv.labs.arxiv.org/html/2101.03201)
- [Kaehler et al. 2012](https://pubmed.ncbi.nlm.nih.gov/26357114/)
- [arXiv:1812.05609](https://arxiv.org/abs/1812.05609)
- [arXiv:1503.05920](https://arxiv.org/pdf/1503.05920)
