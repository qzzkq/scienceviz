"""Блин Зельдовича: газ, гравитация и расширение вместе.

Одна синусоидальная волна в периодическом коробе [−L/2, L/2) схлопывается
в плоскость x = 0 при a = a_c. До схлопывания решение точное (equations.md,
раздел 9). После схлопывания точного решения нет; опорой служит
тёмная материя, посчитанная «листами» — частицами в 1D с точной силой.

Запуск: python3 zeldovich.py — печатает таблицы и рисует figures/pancake_*.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cosmology import Cosmology, internal_energy, temperature
from hydro1d import ENT, Gas, Sim1D

FIGURES = Path(__file__).parent / "figures"

LENGTH = 64.0     # Мпк/h
A_INIT = 0.02     # z = 49
A_CAUSTIC = 1 / 3 # z = 2
A_PRE = 0.25      # z = 3: ещё до схлопывания, есть точное решение
A_OUT = (A_PRE, 0.5, 1.0)
T_INIT = 100.0    # К, начальная температура адиабатического газа
JEANS_CELLS = 2.0 # длина Джинса изотермического газа при a = 1, в ячейках


class Pancake:
    def __init__(self, cosmo, length=LENGTH, a_caustic=A_CAUSTIC):
        self.cosmo, self.length = cosmo, length
        self.k = 2 * np.pi / length
        self.d_caustic = cosmo.growth(a_caustic)

    def amplitude(self, a):
        return self.cosmo.growth(a) / self.d_caustic

    def x_of_q(self, q, a):
        return q - self.amplitude(a) * np.sin(self.k * q) / self.k

    def q_of_x(self, x, a):
        """Обратное отображение (до схлопывания оно монотонно): бисекция."""
        amp, k = self.amplitude(a), self.k
        lo, hi = x - amp / k, x + amp / k
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            below = mid - amp * np.sin(k * mid) / k < x
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        return 0.5 * (lo + hi)

    def velocity(self, q, a):
        """Пекулярная скорость v = a·dx/dt."""
        c = self.cosmo
        return -a * c.hubble(a) * c.growth_rate(a) * self.amplitude(a) * np.sin(self.k * q) / self.k

    def density(self, q, a):
        return 1 / (1 - self.amplitude(a) * np.cos(self.k * q))

    def cell_averages(self, n, a):
        """Точные средние по ячейкам: ρ — масса ячейки, m — её импульс, делённые на dx."""
        c, k, dx = self.cosmo, self.k, self.length / n
        q = self.q_of_x(-self.length / 2 + dx * np.arange(n + 1), a)
        rho = np.diff(q) / dx
        pref = a * c.hubble(a) * c.growth_rate(a) * self.amplitude(a) / k**2
        mom = pref * np.diff(np.cos(k * q)) / dx
        return rho, mom


def isothermal_cs(cosmo, n, length=LENGTH, jeans_cells=JEANS_CELLS):
    """Скорость звука, при которой сопутствующая длина Джинса при a = 1 равна jeans_cells ячейкам."""
    lam = jeans_cells * length / n
    return lam / (2 * np.pi) * np.sqrt(cosmo.four_pi_g)


def make_gas(kind, cosmo, n, dual_energy=True):
    if kind == "isothermal":
        return Gas("isothermal", cs=isothermal_cs(cosmo, n))
    if kind == "adiabatic":
        return Gas("adiabatic", dual_energy=dual_energy)
    return Gas("dust")


def run_gas(pancake, gas, n, order=1, a_out=A_OUT):
    """Прогон газа; возвращает {a: (ρ, v, P, Ŝ или None)} и Sim1D."""
    rho, mom = pancake.cell_averages(n, A_INIT)
    v = mom / rho
    p = (gas.gamma - 1) * rho * internal_energy(T_INIT) if gas.kind == "adiabatic" else gas.cs**2 * rho
    U = gas.conserved(rho, v, p, A_INIT)
    sim = Sim1D(gas, U, pancake.length, order=order, cosmo=pancake.cosmo, t=pancake.cosmo.t_of_a(A_INIT))
    out = {}
    for a in a_out:
        sim.run_to_a(a)
        rho, v, p = sim.primitives()
        out[a] = (rho, v, p, sim.U[ENT].copy() if gas.kind == "adiabatic" else None)
    return out, sim


def run_dark_matter(pancake, n_part=2**16, a_out=A_OUT, dlna=1e-3):
    """Тёмная материя «листами»: частицы в 1D, точная сила по порядку частиц, шаги KDK.

    Сила в 1D зависит только от массы слева: ∂φ/∂x = (4πGρ̄/a)·(M(x) − x) + const.
    Возвращает {a: (q, x, v)}.
    """
    c, L = pancake.cosmo, pancake.length
    q = -L / 2 + (np.arange(n_part) + 0.5) * L / n_part
    x = pancake.x_of_q(q, A_INIT)
    u = A_INIT * pancake.velocity(q, A_INIT)  # u = a·v, du/dt = −∂φ/∂x, dx/dt = u/a²

    def accel(x, a):
        rank = np.empty(n_part)
        rank[np.argsort(x)] = np.arange(n_part)
        grad = c.four_pi_g / a * ((rank + 0.5) * L / n_part - (x + L / 2))
        return -(grad - grad.mean())

    a_grid = np.unique(np.concatenate([np.exp(np.arange(np.log(A_INIT), np.log(max(a_out)), dlna)), a_out]))
    t_grid = c.t_of_a(a_grid)
    acc = accel(x, A_INIT)
    out = {}
    for i in range(len(a_grid) - 1):
        dt = t_grid[i + 1] - t_grid[i]
        a_mid = c.a_of_t(t_grid[i] + dt / 2)
        u += 0.5 * dt * acc
        x = (x + u * dt / a_mid**2 + L / 2) % L - L / 2
        acc = accel(x, a_grid[i + 1])
        u += 0.5 * dt * acc
        if a_grid[i + 1] in a_out:
            out[a_grid[i + 1]] = (q.copy(), x.copy(), u / a_grid[i + 1])
    return out


def dm_density(x, n, length=LENGTH):
    counts, _ = np.histogram(x, bins=n, range=(-length / 2, length / 2))
    return counts * n / len(x)


def outer_caustic(q, x):
    """Граница многопотоковой области: самая дальняя точка поворота x(q)."""
    turn = np.nonzero(np.diff(np.sign(np.diff(x))))[0] + 1
    return np.abs(x[turn]).max() if len(turn) else 0.0


def shocked_edge(rho, ent, x, s_ref):
    """Граница области, где удельная энтропия выросла в 10 раз и больше: газ прошёл ударную волну."""
    hot = ent / rho > 10 * s_ref
    return np.abs(x[hot]).max() if hot.any() else 0.0


def smooth(f, cells):
    kernel = np.ones(cells) / cells
    return np.convolve(np.concatenate([f[-cells:], f, f[:cells]]), kernel, mode="same")[cells:-cells]


def deviation(rho, dm_rho, x, n):
    """Среднее |Δ lg ρ| газа от ТМ в центральных ±8 Мпк/h после сглаживания на 1 Мпк/h."""
    cells = max(1, int(round(1.0 / (LENGTH / n))))
    inner = np.abs(x) < 8
    return np.abs(np.log10(smooth(rho, cells)) - np.log10(smooth(dm_rho, cells)))[inner].mean()


def precaustic_error(out, pancake, n):
    exact, _ = pancake.cell_averages(n, A_PRE)
    return np.abs(out[A_PRE][0] - exact).sum() / exact.sum()


def precaustic_convergence(cosmo, pancake, ns=(64, 128, 256, 512, 1024)):
    """Относительная ошибка L1 плотности при a = A_PRE для всех вариантов и порядков."""
    errors = {}
    for kind in KINDS:
        for order in (1, 2):
            errors[kind, order] = [precaustic_error(run_gas(pancake, make_gas(kind, cosmo, n), n, order,
                                                            a_out=(A_PRE,))[0], pancake, n) for n in ns]
    return errors, ns


def isothermal_sweep(cosmo, pancake, dm, n=256, order=2, cs_kms=(None, 50, 150, 300)):
    """Изотермический газ при разной скорости звука: точность до схлопывания и профиль после."""
    x = -LENGTH / 2 + (np.arange(n) + 0.5) * LENGTH / n
    dm_rho = dm_density(dm[1.0][1], n)
    rows = []
    for cs in cs_kms:
        gas = make_gas("isothermal", cosmo, n) if cs is None else Gas("isothermal", cs=cs / 100)
        out, _ = run_gas(pancake, gas, n, order, a_out=(A_PRE, 1.0))
        jeans = 2 * np.pi * gas.cs / np.sqrt(cosmo.four_pi_g)  # при a = 1
        rho = out[1.0][0]
        rows.append((gas.cs * 100, jeans, precaustic_error(out, pancake, n), rho.max(),
                     rho[np.abs(x) < 1].mean(), deviation(rho, dm_rho, x, n)))
    return rows


KINDS = ("dust", "isothermal", "adiabatic")


def main():
    FIGURES.mkdir(exist_ok=True)
    cosmo = Cosmology()
    pancake = Pancake(cosmo)
    n = 256
    x = -LENGTH / 2 + (np.arange(n) + 0.5) * LENGTH / n
    print(f"A(a_init) = {pancake.amplitude(A_INIT):.4f}, A(a_pre) = {pancake.amplitude(A_PRE):.4f}, "
          f"A(1) = {pancake.amplitude(1.0):.3f}")
    print(f"изотермическая скорость звука при N = {n}: {isothermal_cs(cosmo, n) * 100:.2f} км/с")

    runs, sims = {}, {}
    for kind in KINDS + ("adiabatic-no-dual",):
        for order in (1, 2):
            gas = make_gas("adiabatic", cosmo, n, dual_energy=False) if kind == "adiabatic-no-dual" \
                else make_gas(kind, cosmo, n)
            runs[kind, order], sims[kind, order] = run_gas(pancake, gas, n, order)
    dm = run_dark_matter(pancake)

    print(f"\nN = {n}: шагов до a = 1, дрейф массы, относительная ошибка L1 при z = {1 / A_PRE - 1:.0f}")
    exact_rho, exact_mom = pancake.cell_averages(n, A_PRE)
    for (kind, order), sim in sims.items():
        rho, v = runs[kind, order][A_PRE][:2]
        err_rho = np.abs(rho - exact_rho).sum() / exact_rho.sum()
        err_mom = np.abs(rho * v - exact_mom).sum() / np.abs(exact_mom).sum()
        print(f"  {kind:>17}, порядок {order}: {sim.steps:4d} шагов, масса {abs(sim.mass() / LENGTH - 1):.0e}, "
              f"ρ {err_rho:.1e}, ρv {err_mom:.1e}")
    q_dm, x_dm, _ = dm[A_PRE]
    print(f"  тёмная материя: макс. отклонение положений {np.abs(x_dm - pancake.x_of_q(q_dm, A_PRE)).max():.0e} Мпк/h")

    errors, ns = precaustic_convergence(cosmo, pancake)
    print(f"\nсходимость при z = {1 / A_PRE - 1:.0f}, относительная ошибка L1 плотности, N = {ns}:")
    for (kind, order), err in errors.items():
        rate = -np.polyfit(np.log(ns), np.log(err), 1)[0]
        print(f"  {kind:>10}, порядок {order}: " + " ".join(f"{e:.1e}" for e in err) + f"  (порядок {rate:.2f})")

    print(f"\nпосле схлопывания, N = {n}: макс. ρ, средняя ρ в |x| < 1, граница сжатой области, "
          "отклонение от ТМ")
    for a in A_OUT[1:]:
        q_dm, x_dm, _ = dm[a]
        dm_rho = dm_density(x_dm, n)
        print(f"  z = {1 / a - 1:.0f}  {'тёмная материя':>17}           {dm_rho.max():6.1f} {dm_rho[np.abs(x) < 1].mean():6.1f}"
              f"  {outer_caustic(q_dm, x_dm):5.2f}")
        for order in (1, 2):
            for kind in KINDS:
                rho, _, _, ent = runs[kind, order][a]
                edge = ""
                if kind == "adiabatic":
                    s_ref = (runs[kind, order][A_PRE][3] / runs[kind, order][A_PRE][0]).min()
                    edge = f"{shocked_edge(rho, ent, x, s_ref):5.2f}"
                print(f"  z = {1 / a - 1:.0f}  {kind:>17}, порядок {order} {rho.max():6.1f} "
                      f"{rho[np.abs(x) < 1].mean():6.1f}  {edge:>5}  {deviation(rho, dm_rho, x, n):.3f}")

    print(f"\nизотермический газ, порядок 2, N = {n}: c, км/с | λJ(a=1), Мпк/h | ошибка при z = 3 | "
          "макс. ρ | ρ в |x| < 1 | отклонение от ТМ")
    for row in isothermal_sweep(cosmo, pancake, dm, n):
        print("  {:6.1f} {:6.2f} {:9.1e} {:7.1f} {:6.1f} {:7.3f}".format(*row))

    plot_precaustic(pancake, runs, n, x)
    plot_postcaustic(runs, dm, n, x)
    plot_temperature(runs, x)
    plot_convergence(errors, ns)


COLORS = {"dust": "C0", "isothermal": "C1", "adiabatic": "C2", "adiabatic-no-dual": "C3", "dm": "0.45"}
LABELS = {"dust": "пыль (P = 0)", "isothermal": "изотермический", "adiabatic": "адиабатический",
          "adiabatic-no-dual": "адиабатический без двойной энергии", "dm": "тёмная материя (листы)"}


def plot_precaustic(pancake, runs, n, x):
    xe = np.linspace(-LENGTH / 2, LENGTH / 2, 2001)
    qe = pancake.q_of_x(xe, A_PRE)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for kind, marker in zip(KINDS, "o^s"):
        rho, v = runs[kind, 2][A_PRE][:2]
        axes[0].plot(x, rho, marker, ms=3, mfc="none", color=COLORS[kind], label=f"{LABELS[kind]}, порядок 2")
        axes[1].plot(x, v * 100, marker, ms=3, mfc="none", color=COLORS[kind])
    rho, v = runs["dust", 1][A_PRE][:2]
    axes[0].plot(x, rho, "--", color="C4", lw=1, label="пыль, порядок 1")
    axes[1].plot(x, v * 100, "--", color="C4", lw=1)
    axes[0].plot(xe, pancake.density(qe, A_PRE), "k-", lw=1, label="точное решение")
    axes[1].plot(xe, pancake.velocity(qe, A_PRE) * 100, "k-", lw=1)
    axes[0].set_ylabel("ρ / ρ̄")
    axes[1].set_ylabel("v, км/с")
    for ax in axes:
        ax.set_xlabel("x, Мпк/h")
        ax.set_xlim(-12, 12)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Блин Зельдовича до схлопывания: z = {1 / A_PRE - 1:.0f}, N = {n}")
    fig.tight_layout()
    fig.savefig(FIGURES / "pancake_precaustic.png", dpi=130)
    plt.close(fig)


def plot_postcaustic(runs, dm, n, x, order=2):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for col, a in enumerate(A_OUT[1:]):
        _, x_dm, v_dm = dm[a]
        axes[0, col].semilogy(x, dm_density(x_dm, n), "-", color=COLORS["dm"], lw=2.5, label=LABELS["dm"])
        axes[1, col].plot(x_dm[::8], v_dm[::8] * 100, ".", ms=0.5, color=COLORS["dm"])
        for kind in KINDS:
            rho, v = runs[kind, order][a][:2]
            style = ":" if kind == "isothermal" else "-"
            axes[0, col].semilogy(x, rho, style, color=COLORS[kind], lw=1.2, label=LABELS[kind])
            axes[1, col].plot(x, v * 100, style, color=COLORS[kind], lw=1.2)
        axes[0, col].set_title(f"z = {1 / a - 1:.0f}")
        axes[1, col].set_xlabel("x, Мпк/h")
        axes[1, col].set_xlim(-12, 12)
    axes[0, 0].set_ylabel("ρ / ρ̄")
    axes[1, 0].set_ylabel("v, км/с")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"После схлопывания (z_c = {1 / A_CAUSTIC - 1:.0f}), N = {n}, порядок {order}. "
                 "Серые точки внизу — фазовое пространство ТМ; изотермический совпал с пылью")
    fig.tight_layout()
    fig.savefig(FIGURES / "pancake_postcaustic.png", dpi=130)
    plt.close(fig)


def plot_temperature(runs, x, order=2):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, a in zip(axes, (A_PRE, 1.0)):
        for kind in ("adiabatic", "adiabatic-no-dual"):
            rho, _, p = runs[kind, order][a][:3]
            ax.semilogy(x, temperature(p / (2 / 3 * rho)), "-", color=COLORS[kind], lw=1, label=LABELS[kind])
        rho = runs["adiabatic", order][a][0]
        ax.semilogy(x, T_INIT * rho ** (2 / 3) * (A_INIT / a) ** 2, "k:", lw=1,
                    label="адиабатическое остывание без ударных волн")
        ax.set_title(f"z = {1 / a - 1:.0f}")
        ax.set_xlabel("x, Мпк/h")
    axes[0].set_ylabel("T, К")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Температура адиабатического газа, порядок {order}: без двойной энергии — ложный нагрев")
    fig.tight_layout()
    fig.savefig(FIGURES / "pancake_temperature.png", dpi=130)
    plt.close(fig)


def plot_convergence(errors, ns):
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for (kind, order), err in errors.items():
        ax.loglog(ns, err, "o-" if order == 1 else "s--", color=COLORS[kind], label=f"{LABELS[kind]}, порядок {order}")
    ax.set_xlabel("число ячеек N")
    ax.set_ylabel("относительная ошибка L1 плотности")
    ax.set_title(f"Блин до схлопывания, z = {1 / A_PRE - 1:.0f}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "pancake_convergence.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
