"""Проверки 3D-снимков: сохранение массы, рост возмущений, спектры.

Функции принимают обычные массивы n³, поэтому годятся и для снимков C++-ядра,
когда будет решён их формат: достаточно прочитать плотность в numpy.
"""
import numpy as np

from ics import measure_power


def mass_drift(rho):
    """Относительное отклонение средней плотности от 1 (единицы кода: ρ̄ = 1)."""
    return abs(rho.mean() - 1)


def growth_ratio(delta_start, a_start, delta, a, length, cosmo, kmax=None):
    """P(k, a) / [P(k, a_нач)·(D(a)/D(a_нач))²] по слоям k.

    На больших масштабах должно быть 1: рост совпадает с линейной теорией.
    Сравнение мода к моде одного и того же поля, поэтому космической дисперсии нет.
    """
    k, p0, _, count = measure_power(delta_start, length)
    _, p1, _, _ = measure_power(delta, length)
    ratio = p1 / p0 / (cosmo.growth(a) / cosmo.growth(a_start)) ** 2
    keep = k <= kmax if kmax is not None else slice(None)
    return k[keep], ratio[keep], count[keep]


def cross_correlation(d1, d2, length):
    """Коэффициент корреляции r(k) = P12 / sqrt(P11·P22) по слоям k: 1 — поля совпадают с точностью до множителя."""
    n = d1.shape[0]
    f1, f2 = np.fft.fftn(d1), np.fft.fftn(d2)
    k1 = 2 * np.pi * np.fft.fftfreq(n, length / n)
    kk = np.sqrt(k1[:, None, None] ** 2 + k1[None, :, None] ** 2 + k1[None, None, :] ** 2)
    bins = np.floor(kk / (2 * np.pi / length) + 0.5).astype(int).ravel()
    keep = (bins >= 1) & (bins <= n // 2)
    b = bins[keep]

    def binned(x):
        return np.bincount(b, x.ravel()[keep], n // 2 + 1)[1:]

    count = np.bincount(b, minlength=n // 2 + 1)[1:]
    k = binned(kk) / count
    r = binned((f1 * f2.conj()).real) / np.sqrt(binned(np.abs(f1) ** 2) * binned(np.abs(f2) ** 2))
    return k, r


def mass_fraction_above(rho, thresholds):
    """Доля массы в ячейках с ρ выше порога."""
    total = rho.sum()
    return [rho[rho > t].sum() / total for t in thresholds]
