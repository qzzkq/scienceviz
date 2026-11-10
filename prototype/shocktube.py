"""Ударная труба: сверка схемы газа с точным решением задачи Римана.

Запуск: python3 shocktube.py — печатает ошибки и рисует figures/shocktube_*.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hydro1d import Gas, Sim1D
from riemann import ideal_sample, isothermal_sample

FIGURES = Path(__file__).parent / "figures"
X0, T_END = 0.5, 0.2

# Сод (1978): классический тест для идеального газа
SOD = dict(left=(1.0, 0.0, 1.0), right=(0.125, 0.0, 0.1), gamma=1.4)
# изотермический аналог: перепад плотности в 10 раз
ISO = dict(left=(1.0, 0.0), right=(0.1, 0.0), cs=1.0)


def run(test, n, order):
    """Возвращает центры ячеек и Sim1D в момент T_END."""
    x = (np.arange(n) + 0.5) / n
    left = x < X0
    if test == "sod":
        gas = Gas("adiabatic", gamma=SOD["gamma"], dual_energy=False)
        rho = np.where(left, SOD["left"][0], SOD["right"][0])
        p = np.where(left, SOD["left"][2], SOD["right"][2])
    else:
        gas = Gas("isothermal", cs=ISO["cs"])
        rho = np.where(left, ISO["left"][0], ISO["right"][0])
        p = ISO["cs"] ** 2 * rho
    sim = Sim1D(gas, gas.conserved(rho, np.zeros(n), p, 1.0), 1.0, order=order, bc="outflow")
    sim.run_to_time(T_END)
    return x, sim


def exact(test, x):
    xi = (x - X0) / T_END
    if test == "sod":
        return ideal_sample(SOD["left"], SOD["right"], SOD["gamma"], xi)
    return isothermal_sample(ISO["left"], ISO["right"], ISO["cs"], xi)


def exact_cell_density(test, n, sub=16):
    """Точная плотность, усреднённая по ячейке."""
    x = (np.arange(n * sub) + 0.5) / (n * sub)
    return exact(test, x)[0].reshape(n, sub).mean(axis=1)


def l1_error(test, n, order):
    _, sim = run(test, n, order)
    return np.abs(sim.U[0] - exact_cell_density(test, n)).mean()


def convergence(ns=(50, 100, 200, 400, 800, 1600)):
    return {(test, order): [l1_error(test, n, order) for n in ns]
            for test in ("sod", "iso") for order in (1, 2)}, ns


def plot_profiles(test, n=200):
    xe = np.linspace(0, 1, 2001)
    ex = exact(test, xe)
    names = ["плотность", "скорость", "давление"] if test == "sod" else ["плотность", "скорость"]
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.6))
    for order, style in ((1, "o"), (2, "s")):
        x, sim = run(test, n, order)
        num = sim.primitives()
        for i, ax in enumerate(axes):
            ax.plot(x, num[i], style, ms=2.5, mfc="none", label=f"порядок {order}, N = {n}")
    for i, ax in enumerate(axes):
        ax.plot(xe, ex[i], "k-", lw=1, label="точное решение")
        ax.set_title(names[i])
        ax.set_xlabel("x")
    axes[0].legend(fontsize=8)
    title = "Ударная труба Сода, γ = 1.4" if test == "sod" else "Изотермическая ударная труба, c = 1"
    fig.suptitle(f"{title}, t = {T_END}, HLL")
    fig.tight_layout()
    fig.savefig(FIGURES / f"shocktube_{test}.png", dpi=130)
    plt.close(fig)


def plot_convergence(errors, ns):
    fig, ax = plt.subplots(figsize=(5.2, 4))
    labels = {"sod": "Сод", "iso": "изотермическая"}
    for (test, order), err in errors.items():
        ax.loglog(ns, err, "o-" if order == 1 else "s--", label=f"{labels[test]}, порядок {order}")
    ns_arr = np.array(ns, dtype=float)
    ax.loglog(ns_arr, 0.6 * ns_arr ** -0.5, "k:", lw=1, label="∝ N^-1/2")
    ax.loglog(ns_arr, 1.5 * ns_arr ** -1.0, "k-.", lw=1, label="∝ N^-1")
    ax.set_xlabel("число ячеек N")
    ax.set_ylabel("ошибка плотности L1")
    ax.legend(fontsize=8)
    ax.set_title("Сходимость к точному решению")
    fig.tight_layout()
    fig.savefig(FIGURES / "shocktube_convergence.png", dpi=130)
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    plot_profiles("sod")
    plot_profiles("iso")
    errors, ns = convergence()
    plot_convergence(errors, ns)
    print("ошибка плотности L1:")
    print("N      " + "  ".join(f"{t}/{o}".rjust(9) for t, o in errors))
    for i, n in enumerate(ns):
        print(f"{n:<6} " + "  ".join(f"{err[i]:9.2e}" for err in errors.values()))
    for key, err in errors.items():
        rate = np.polyfit(np.log(ns), np.log(err), 1)[0]
        print(f"{key}: порядок сходимости {-rate:.2f}")


if __name__ == "__main__":
    main()
