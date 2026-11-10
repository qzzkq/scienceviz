"""Начальные условия: гауссово поле по спектру ΛCDM и скорости по линейной теории.

Алгоритм описан в equations.md, раздел 8; C++-версия должна повторять его
шаг в шаг, чтобы поля совпадали при одном зерне генератора.

Запуск: python3 ics.py — проверяет спектр сгенерированного поля и рисует
figures/ics_*.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cosmology import Cosmology

FIGURES = Path(__file__).parent / "figures"
T_CMB = 2.7255


def transfer_eh_nowiggle(k, cosmo):
    """Передаточная функция Эйзенштейна–Ху без барионных осцилляций (EH98, ур. 26–31). k в h/Мпк."""
    om, h = cosmo.omega_m, cosmo.h
    omh2, obh2, fb = om * h * h, cosmo.omega_b * h * h, cosmo.omega_b / om
    theta = T_CMB / 2.7
    s = 44.5 * np.log(9.83 / omh2) / np.sqrt(1 + 10 * obh2**0.75)  # звуковой горизонт, Мпк
    alpha = 1 - 0.328 * np.log(431 * omh2) * fb + 0.38 * np.log(22.3 * omh2) * fb**2
    gamma_eff = om * h * (alpha + (1 - alpha) / (1 + (0.43 * k * h * s) ** 4))
    q = k * theta**2 / gamma_eff
    l0 = np.log(2 * np.e + 1.8 * q)
    c0 = 14.2 + 731 / (1 + 62.5 * q)
    return l0 / (l0 + c0 * q * q)


def transfer_eh(k, cosmo):
    """Полная передаточная функция Эйзенштейна–Ху с барионными осцилляциями (EH98, ур. 2–24). k в h/Мпк."""
    om, h = cosmo.omega_m, cosmo.h
    wm, wb = om * h * h, cosmo.omega_b * h * h
    fb = cosmo.omega_b / om
    fc = 1 - fb
    theta = T_CMB / 2.7
    k = np.asarray(k, dtype=float) * h  # дальше всё в 1/Мпк

    z_eq = 2.50e4 * wm * theta**-4
    k_eq = 7.46e-2 * wm * theta**-2
    b1 = 0.313 * wm**-0.419 * (1 + 0.607 * wm**0.674)
    b2 = 0.238 * wm**0.223
    z_d = 1291 * wm**0.251 / (1 + 0.659 * wm**0.828) * (1 + b1 * wb**b2)
    r_d = 31.5 * wb * theta**-4 * (1e3 / z_d)
    r_eq = 31.5 * wb * theta**-4 * (1e3 / z_eq)
    s = 2 / (3 * k_eq) * np.sqrt(6 / r_eq) * np.log((np.sqrt(1 + r_d) + np.sqrt(r_d + r_eq)) / (1 + np.sqrt(r_eq)))
    k_silk = 1.6 * wb**0.52 * wm**0.73 * (1 + (10.4 * wm) ** -0.95)
    q = k / (13.41 * k_eq)

    def t0(alpha, beta):
        lg = np.log(np.e + 1.8 * beta * q)
        return lg / (lg + (14.2 / alpha + 386 / (1 + 69.9 * q**1.08)) * q * q)

    # холодная тёмная материя
    a1 = (46.9 * wm) ** 0.670 * (1 + (32.1 * wm) ** -0.532)
    a2 = (12.0 * wm) ** 0.424 * (1 + (45.0 * wm) ** -0.582)
    alpha_c = a1**-fb * a2 ** -(fb**3)
    bb1 = 0.944 / (1 + (458 * wm) ** -0.708)
    bb2 = (0.395 * wm) ** -0.0266
    beta_c = 1 / (1 + bb1 * (fc**bb2 - 1))
    f = 1 / (1 + (k * s / 5.4) ** 4)
    t_cdm = f * t0(1, beta_c) + (1 - f) * t0(alpha_c, beta_c)

    # барионы
    y = (1 + z_eq) / (1 + z_d)
    g = y * (-6 * np.sqrt(1 + y) + (2 + 3 * y) * np.log((np.sqrt(1 + y) + 1) / (np.sqrt(1 + y) - 1)))
    alpha_b = 2.07 * k_eq * s * (1 + r_d) ** -0.75 * g
    beta_node = 8.41 * wm**0.435
    beta_b = 0.5 + fb + (3 - 2 * fb) * np.sqrt((17.2 * wm) ** 2 + 1)
    s_tilde = s / (1 + (beta_node / (k * s)) ** 3) ** (1 / 3)
    t_b = (t0(1, 1) / (1 + (k * s / 5.2) ** 2)
           + alpha_b / (1 + (beta_b / (k * s)) ** 3) * np.exp(-((k / k_silk) ** 1.4))) * np.sinc(k * s_tilde / np.pi)
    return fb * t_b + fc * t_cdm


def sound_horizon_eh(cosmo):
    """Звуковой горизонт на эпоху отрыва барионов, Мпк (EH98, ур. 6)."""
    wm, wb = cosmo.omega_m * cosmo.h**2, cosmo.omega_b * cosmo.h**2
    theta = T_CMB / 2.7
    z_eq, k_eq = 2.50e4 * wm * theta**-4, 7.46e-2 * wm * theta**-2
    b1 = 0.313 * wm**-0.419 * (1 + 0.607 * wm**0.674)
    z_d = 1291 * wm**0.251 / (1 + 0.659 * wm**0.828) * (1 + b1 * wb ** (0.238 * wm**0.223))
    r_d, r_eq = 31.5 * wb * theta**-4 * (1e3 / z_d), 31.5 * wb * theta**-4 * (1e3 / z_eq)
    return 2 / (3 * k_eq) * np.sqrt(6 / r_eq) * np.log((np.sqrt(1 + r_d) + np.sqrt(r_d + r_eq)) / (1 + np.sqrt(r_eq)))


def top_hat(x):
    return 3 * (np.sin(x) - x * np.cos(x)) / x**3


class LinearPower:
    """Линейный спектр мощности P(k, a) в (Мпк/h)³, нормированный на σ8 при a = 1.

    wiggles=True — с барионными осцилляциями (полная формула EH98).
    """

    def __init__(self, cosmo, wiggles=False):
        self.cosmo = cosmo
        self.transfer = transfer_eh if wiggles else transfer_eh_nowiggle
        self.d1 = cosmo.growth(1.0)
        self.amplitude = 1.0
        self.amplitude = (cosmo.sigma8 / self.sigma(8.0)) ** 2

    def __call__(self, k, a=1.0):
        k = np.asarray(k, dtype=float)
        growth = (self.cosmo.growth(a) / self.d1) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            p = self.amplitude * k**self.cosmo.n_s * self.transfer(k, self.cosmo) ** 2
        return np.where(k > 0, p, 0.0) * growth

    def sigma(self, r, a=1.0):
        """Среднеквадратичная δ в шаре радиуса r Мпк/h."""
        k = np.logspace(-5, 2, 20000)
        integrand = k**3 * self(k, a) * top_hat(k * r) ** 2 / (2 * np.pi**2)
        return np.sqrt(np.trapezoid(integrand, np.log(k)))


def wavevectors(n, length):
    """Компоненты волнового вектора для rfftn-раскладки сетки n³."""
    k = 2 * np.pi * np.fft.fftfreq(n, length / n)
    kz = 2 * np.pi * np.fft.rfftfreq(n, length / n)
    return k[:, None, None], k[None, :, None], kz[None, None, :]


def gaussian_field(n, length, power, a, seed):
    """Фурье-образ (rfftn) поля δ на сетке n³ со спектром power(k, a).

    Белый шум в обычном пространстве, его БПФ умножается на sqrt(P(k)/dx³):
    так поле сразу вещественное, а одно зерно даёт одно поле при любом числе процессов.
    Моды Найквиста обнуляются: иначе скорость i·k/k²·δ на них не вещественна.
    """
    dx = length / n
    noise = np.random.default_rng(seed).standard_normal((n, n, n))
    kx, ky, kz = wavevectors(n, length)
    kk = np.sqrt(kx**2 + ky**2 + kz**2)
    delta_k = np.fft.rfftn(noise) * np.sqrt(power(kk, a) / dx**3)
    delta_k[0, 0, 0] = 0.0
    nyq = n // 2
    delta_k[nyq, :, :] = delta_k[:, nyq, :] = delta_k[:, :, nyq] = 0.0
    return delta_k


def linear_velocity(delta_k, n, length, cosmo, a):
    """Пекулярная скорость растущей моды: v_k = i·a·H·f·k/k²·δ_k, в единицах 100 км/с."""
    kx, ky, kz = wavevectors(n, length)
    k2 = kx**2 + ky**2 + kz**2
    k2[0, 0, 0] = 1.0
    factor = 1j * a * cosmo.hubble(a) * cosmo.growth_rate(a) * delta_k / k2
    return [np.fft.irfftn(factor * kc, s=(n, n, n), axes=(0, 1, 2)) for kc in (kx, ky, kz)]


def initial_conditions(n, length, cosmo, a, seed):
    """Плотность (ρ/ρ̄ = 1 + δ) и три компоненты скорости на сетке n³."""
    power = LinearPower(cosmo)
    delta_k = gaussian_field(n, length, power, a, seed)
    delta = np.fft.irfftn(delta_k, s=(n, n, n), axes=(0, 1, 2))
    return 1 + delta, linear_velocity(delta_k, n, length, cosmo, a)


def measure_power(delta, length, power=None, a=1.0):
    """Спектр поля по сферическим слоям шириной k_f = 2π/L.

    Возвращает k, измеренный P, ожидаемый P (усреднённый по тем же модам), число мод.
    """
    n = delta.shape[0]
    dx = length / n
    p_modes = np.abs(np.fft.fftn(delta)) ** 2 * dx**3 / n**3
    k1 = 2 * np.pi * np.fft.fftfreq(n, dx)
    kk = np.sqrt(k1[:, None, None] ** 2 + k1[None, :, None] ** 2 + k1[None, None, :] ** 2)
    kf = 2 * np.pi / length
    bins = np.floor(kk / kf + 0.5).astype(int).ravel()
    nb = n // 2
    keep = (bins >= 1) & (bins <= nb)
    count = np.bincount(bins[keep], minlength=nb + 1)[1:]
    k_mean = np.bincount(bins[keep], kk.ravel()[keep], nb + 1)[1:] / count
    p_mean = np.bincount(bins[keep], p_modes.ravel()[keep], nb + 1)[1:] / count
    p_expected = None
    if power is not None:
        # моды Найквиста в поле обнулены, в ожидании тоже
        on_grid = (np.abs(k1) < np.pi / dx * (1 - 1e-9)).astype(float)
        mask = (on_grid[:, None, None] * on_grid[None, :, None] * on_grid[None, None, :]).ravel()
        p_expected = np.bincount(bins[keep], (power(kk.ravel(), a) * mask)[keep], nb + 1)[1:] / count
    return k_mean, p_mean, p_expected, count


def main():
    FIGURES.mkdir(exist_ok=True)
    cosmo = Cosmology()
    power = LinearPower(cosmo)
    print(f"σ8 = {power.sigma(8.0):.4f} (задано {cosmo.sigma8}), P(k = 0.1 h/Мпк, z = 0) = {power(0.1):.0f} (Мпк/h)³")

    n, length, a_init, seed = 128, 256.0, 0.02, 42
    rho, vel = initial_conditions(n, length, cosmo, a_init, seed)
    delta = rho - 1
    disp = np.sqrt(np.mean(sum(v**2 for v in vel))) / (a_init * cosmo.hubble(a_init) * cosmo.growth_rate(a_init))
    print(f"N = {n}³, L = {length:.0f} Мпк/h, z = {1 / a_init - 1:.0f}: δ_rms = {delta.std():.4f}, "
          f"min ρ = {rho.min():.3f}, v_rms = {np.sqrt(np.mean(sum(v**2 for v in vel))) * 100:.1f} км/с, "
          f"смещение_rms = {disp:.3f} Мпк/h = {disp / (length / n):.2f} ячейки")

    k, p, p_exp, count = measure_power(delta, length, power, a_init)
    good = count >= 100
    ratio = p[good] / p_exp[good]
    # моды k и −k одного поля не независимы, поэтому независимых мод в слое count / 2
    print(f"измеренный P / ожидаемый по слоям с ≥ 100 модами: среднее {ratio.mean():.3f}, "
          f"разброс {ratio.std():.3f} (космическая дисперсия ~ {np.mean(np.sqrt(4 / count[good])):.3f})")

    # дивергенция скорости должна быть −a·H·f·δ
    kx, ky, kz = wavevectors(n, length)
    div_k = sum(1j * kc * np.fft.rfftn(v) for kc, v in zip((kx, ky, kz), vel))
    div = np.fft.irfftn(div_k, s=(n, n, n), axes=(0, 1, 2))
    target = -a_init * cosmo.hubble(a_init) * cosmo.growth_rate(a_init) * delta
    print(f"|∇·v + aHfδ| / |aHfδ| = {np.abs(div - target).max() / np.abs(target).max():.1e}")

    plot(power, k, p, p_exp, count, delta, length, a_init)


def plot(power, k, p, p_exp, count, delta, length, a_init):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    kk = np.logspace(-3, 1, 400)
    growth = power(1.0, 1.0) / power(1.0, a_init)  # пересчёт к z = 0
    axes[0].loglog(kk, power(kk), "k-", lw=1, label="линейный ΛCDM, Эйзенштейн–Ху")
    axes[0].loglog(k, p * growth, "o", ms=3, mfc="none", label="измерен по полю (пересчитан к z = 0)")
    err = p_exp * growth * np.sqrt(4 / count)
    axes[0].fill_between(k, p_exp * growth - err, p_exp * growth + err, color="C1", alpha=0.3,
                         label="ожидание ± космическая дисперсия")
    n = delta.shape[0]
    for kv, name in ((2 * np.pi / length, "k короба"), (np.pi * n / length, "k Найквиста")):
        axes[0].axvline(kv, color="0.6", ls=":", lw=1)
        axes[0].text(kv * 1.08, 1.3e1, name, rotation=90, fontsize=8, va="bottom", ha="left")
    axes[0].set_xlabel("k, h/Мпк")
    axes[0].set_ylabel("P(k), (Мпк/h)³")
    axes[0].set_ylim(1e1, 1e5)
    axes[0].legend(fontsize=8, loc="lower left")
    axes[0].set_title("Спектр мощности начального поля")
    im = axes[1].imshow(delta[:, :, n // 2].T, origin="lower", extent=(0, length, 0, length), cmap="RdBu_r",
                        vmin=-4 * delta.std(), vmax=4 * delta.std())
    fig.colorbar(im, ax=axes[1], label="δ")
    axes[1].set_xlabel("x, Мпк/h")
    axes[1].set_ylabel("y, Мпк/h")
    axes[1].set_title(f"Срез δ при z = {1 / a_init - 1:.0f}, {n}³")
    fig.tight_layout()
    fig.savefig(FIGURES / "ics_power.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
