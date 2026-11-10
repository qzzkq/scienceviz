"""Паутина в 3D: три газа и тёмная материя из одних начальных условий.

    python3 web3d.py run 64 64        прогон: сетка 64³, куб 64 Мпк/h → output/web3d_64.npz
    python3 web3d.py plot 64          проверки и графики по сохранённому прогону

Газ считается hydro3d.py, тёмная материя — particle-mesh из pm3d.py: частиц
столько же, сколько ячеек, сетка сил вдвое мельче. Проверки — из checks.py.
"""
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as fft

from checks import cross_correlation, growth_ratio, mass_fraction_above
from cosmology import Cosmology, internal_energy, temperature
from hydro1d import Gas
from hydro3d import ENT, Gas3D, Sim3D
from ics import LinearPower, gaussian_field, linear_velocity, measure_power
from pm3d import ParticleMesh, zeldovich_particles
from zeldovich import JEANS_CELLS, isothermal_cs

HERE = Path(__file__).parent
OUTPUT, FIGURES = HERE / "output", HERE / "figures"
A_INIT, T_INIT, SEED = 0.02, 100.0, 42
A_OUT = (0.1, 0.25, 0.5, 1.0)
KINDS = ("dust", "isothermal", "adiabatic")
LABELS = {"dm": "тёмная материя (PM)", "dust": "пыль", "isothermal": "изотермический",
          "adiabatic": "адиабатический"}
COLORS = {"dm": "0.35", "dust": "C0", "isothermal": "C1", "adiabatic": "C2"}


def key(name, a):
    return f"{name}_a{a:.3f}"


def initial_fields(n, length, cosmo, scale=1.0):
    """Начальные δ_k, δ и скорость; scale умножает спектр (для проверки в линейном режиме)."""
    power = LinearPower(cosmo)
    delta_k = gaussian_field(n, length, lambda k, a: scale * power(k, a), A_INIT, SEED)
    delta = fft.irfftn(delta_k, s=(n, n, n), workers=-1)
    return delta_k, delta, np.array(linear_velocity(delta_k, n, length, cosmo, A_INIT))


def run_dark_matter(delta_k, n, length, cosmo, out, prefix="dm", dtype=np.float32):
    """Частиц (2n)³ и сетка сил (2n)³ из тех же мод; плотность раскладывается на сетку газа n³.

    При равных сетке и числе частиц частицы отстают от точного решения на 0,2 ячейки,
    а плотность в пустотах — сплошной шум из пустых ячеек.
    """
    pm, grid = ParticleMesh(2 * n, length, cosmo), ParticleMesh(n, length, cosmo)
    pos, u = zeldovich_particles(delta_k, n, length, cosmo, A_INIT, n_part=2 * n)
    out[f"{prefix}_init"] = grid.density(pos).astype(dtype)
    pm.run(pos, u, A_INIT, A_OUT,
           callback=lambda a, pos, u: out.__setitem__(key(prefix, a), grid.density(pos).astype(dtype)))


def run_gas(kind, delta, vel, n, length, cosmo, out, prefix=None, dtype=np.float32):
    gas = Gas("isothermal", cs=isothermal_cs(cosmo, n, length, JEANS_CELLS)) if kind == "isothermal" else Gas(kind)
    rho = 1 + delta
    p = (gas.gamma - 1) * rho * internal_energy(T_INIT) if kind == "adiabatic" else gas.cs**2 * rho
    sim = Sim3D(gas, Gas3D(gas).conserved(rho, vel, p, A_INIT), length / n, cosmo, cosmo.t_of_a(A_INIT))
    prefix = prefix or kind
    drift = []
    for a in A_OUT:
        sim.run_to_a(a)
        rho, _, p = sim.primitives()
        out[key(prefix, a)] = rho.astype(dtype)
        drift.append(abs(sim.mass() / length**3 - 1))
        if kind == "adiabatic":
            out[key("temperature", a)] = temperature(p / (2 / 3 * rho)).astype(np.float32)
    out[f"{prefix}_mass_drift"] = np.array(drift)
    return sim.steps


def run(n, length, n_linear=32):
    OUTPUT.mkdir(exist_ok=True)
    cosmo = Cosmology()
    delta_k, delta, vel = initial_fields(n, length, cosmo)
    out = {"delta_init": delta.astype(np.float32), "length": length, "a_out": np.array(A_OUT)}
    log = []

    t0 = time.time()
    run_dark_matter(delta_k, n, length, cosmo, out)
    log.append(f"тёмная материя, {2 * n}³ частиц: {time.time() - t0:.0f} с")
    print(log[-1], flush=True)
    for kind in KINDS:
        t0 = time.time()
        steps = run_gas(kind, delta, vel, n, length, cosmo, out)
        log.append(f"{kind}: {steps} шагов, {time.time() - t0:.0f} с")
        print(log[-1], flush=True)

    # то же в линейном режиме: спектр × 10⁻⁸, рост обязан совпасть с D(a)
    t0 = time.time()
    lin = {}
    delta_k, delta, vel = initial_fields(n_linear, length, cosmo, scale=1e-8)
    lin["delta_init"] = delta
    run_dark_matter(delta_k, n_linear, length, cosmo, lin, dtype=np.float64)
    run_gas("dust", delta, vel, n_linear, length, cosmo, lin, dtype=np.float64)
    for k, v in lin.items():
        out[f"lin_{k}"] = v
    log.append(f"линейный режим, {n_linear}³: {time.time() - t0:.0f} с")
    print(log[-1], flush=True)
    out["log"] = np.array(log)
    np.savez_compressed(OUTPUT / f"web3d_{n}.npz", **out)


def slab(field, cells=4):
    """Средний lg поля в слое толщиной cells ячеек посередине куба."""
    n = field.shape[2]
    mean = field[:, :, n // 2 - cells // 2:n // 2 + cells // 2].mean(axis=2)
    return np.log10(np.maximum(mean, 1e-3))


def linear_growth(d, length, cosmo):
    """Рост в линейном режиме: {(вещество, a): (k / k_Найквиста, отношение)}."""
    n = d["lin_delta_init"].shape[0]
    out = {}
    for name in ("dm", "dust"):
        start = d["lin_dm_init"] - 1 if name == "dm" else d["lin_delta_init"]
        for a in A_OUT:
            k, ratio, _ = growth_ratio(start, A_INIT, d[f"lin_{key(name, a)}"] - 1, a, length, cosmo)
            out[name, a] = (k / (np.pi * n / length), ratio)
    return out


def plot(n):
    d = np.load(OUTPUT / f"web3d_{n}.npz")
    cosmo = Cosmology()
    length = float(d["length"])
    names = ("dm",) + KINDS
    print("\n".join(d["log"]))

    print("\nсохранение массы газа, |⟨ρ⟩ − 1| при z = 9, 3, 1, 0:")
    for kind in KINDS:
        print(f"  {kind:>11}: " + " ".join(f"{x:.0e}" for x in d[f"{kind}_mass_drift"]))

    lin = linear_growth(d, length, cosmo)
    n_lin = d["lin_delta_init"].shape[0]
    print(f"\nлинейный режим ({n_lin}³): P(k, a) / [P(k, a_нач)·(D/D_нач)²] при z = 0 по k / k_Найквиста")
    kn, _ = lin["dust", 1.0]
    picks = sorted({int(np.argmin(np.abs(kn - x))) for x in (0.1, 0.2, 0.3, 0.4, 0.5)})
    print("  k/kN:        " + "  ".join(f"{kn[i]:.2f}" for i in picks))
    for name in ("dust", "dm"):
        print(f"  {name:>11}: " + "  ".join(f"{lin[name, 1.0][1][i]:.3f}" for i in picks))

    kf = 2 * np.pi / length
    print(f"\nрост в полном прогоне, три первых слоя k = {kf:.3f}, {2 * kf:.3f}, {3 * kf:.3f} h/Мпк:")
    for name in ("dm", "adiabatic", "dust"):
        start = d["dm_init"] if name == "dm" else d["delta_init"] + 1
        for a in A_OUT:
            _, ratio, _ = growth_ratio(start.astype(float) - 1, A_INIT, d[key(name, a)].astype(float) - 1, a,
                                       length, cosmo)
            print(f"  {name:>9}, z = {1 / a - 1:3.0f}: " + "  ".join(f"{r:.3f}" for r in ratio[:3]))

    print("\nz = 0: макс. ρ | доля массы при ρ > 10 и > 100 | r(k) с ТМ при k = 0.2, 0.5, 1 h/Мпк")
    dm = d[key("dm", 1.0)].astype(float)
    for name in names:
        rho = d[key(name, 1.0)].astype(float)
        f10, f100 = mass_fraction_above(rho, (10, 100))
        k, r = cross_correlation(rho - 1, dm - 1, length)
        rk = [np.interp(kv, k, r) for kv in (0.2, 0.5, 1.0)]
        print(f"  {name:>11}: {rho.max():7.0f} | {f10:.2f} {f100:.2f} | " + " ".join(f"{x:.3f}" for x in rk))
    t = d[key("temperature", 1.0)].astype(float)
    rho = d[key("adiabatic", 1.0)].astype(float)
    hot = t > 1e5
    print(f"\nадиабатический газ при z = 0: горячее 10⁵ К {rho[hot].sum() / rho.sum():.0%} массы "
          f"и {hot.mean():.1%} объёма; T от {t.min():.1e} до {t.max():.1e} К")

    plot_slices(d, n, length)
    plot_spectra(d, n, length, cosmo, lin)


def plot_slices(d, n, length):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    extent = (0, length, 0, length)
    for ax, name in zip(axes.flat, ("dm", "dust", "adiabatic", "isothermal")):
        im = ax.imshow(slab(d[key(name, 1.0)].astype(float)).T, origin="lower", extent=extent,
                       cmap="magma", vmin=-1, vmax=2)
        ax.set_title(f"{LABELS[name]}, z = 0")
    fig.colorbar(im, ax=axes[0, :], shrink=0.8, label="lg ρ/ρ̄, слой 4 ячейки")
    t = d[key("temperature", 1.0)].astype(float)
    n2 = n // 2
    im = axes[1, 1].imshow(np.log10(t[:, :, n2 - 2:n2 + 2].mean(axis=2)).T, origin="lower", extent=extent,
                           cmap="inferno", vmin=-2, vmax=7.5)
    axes[1, 1].set_title("адиабатический: температура, z = 0")
    fig.colorbar(im, ax=axes[1, 1], shrink=0.8, label="lg T, К")
    for ax in axes.flat[:5]:
        ax.set_xlabel("x, Мпк/h")
        ax.set_ylabel("y, Мпк/h")
    ax = axes[1, 2]
    bins = np.linspace(-2, 4, 61)
    for name in ("dm",) + KINDS:
        rho = d[key(name, 1.0)].astype(float).ravel()
        h, _ = np.histogram(np.log10(np.maximum(rho, 1e-3)), bins=bins, weights=rho / rho.sum())
        ax.step(bins[:-1], h / np.diff(bins), where="post", color=COLORS[name], label=LABELS[name],
                ls=":" if name == "isothermal" else "-")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1)
    ax.set_xlabel("lg ρ/ρ̄")
    ax.set_ylabel("доля массы на единицу lg ρ")
    ax.set_title("распределение массы по плотности, z = 0")
    ax.legend(fontsize=8)
    fig.suptitle(f"Паутина в 3D: {n}³, куб {length:.0f} Мпк/h, одни начальные условия")
    fig.savefig(FIGURES / f"web3d_slices_{n}.png", dpi=110)
    plt.close(fig)


def plot_spectra(d, n, length, cosmo, lin):
    power = LinearPower(cosmo)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    kk = np.logspace(np.log10(2 * np.pi / length) - 0.1, np.log10(np.pi * n / length), 200)
    ax = axes[0]
    ax.loglog(kk, power(kk), "k--", lw=1, label="линейная теория")
    for name in ("dm",) + KINDS:
        k, p, _, _ = measure_power(d[key(name, 1.0)].astype(float) - 1, length)
        ax.loglog(k, p, color=COLORS[name], label=LABELS[name], ls=":" if name == "isothermal" else "-")
    ax.set_xlabel("k, h/Мпк")
    ax.set_ylabel("P(k), (Мпк/h)³")
    ax.set_title("спектр мощности, z = 0")
    ax.legend(fontsize=8)

    ax = axes[1]
    n_lin = d["lin_delta_init"].shape[0]
    for a, style in ((0.1, "--"), (1.0, "-")):
        for name, label in (("dust", "газ"), ("dm", "тёмная материя (PM)")):
            kn, ratio = lin[name, a]
            ax.plot(kn, ratio, style, color=COLORS[name], label=f"{label}, z = {1 / a - 1:.0f}")
    ax.axhline(1, color="k", lw=0.8)
    ax.axhspan(0.95, 1.05, color="0.9")
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0.6, 1.1)
    ax.set_xlabel("k / k_Найквиста")
    ax.set_ylabel("P(k, a) / [P(k, a_нач)·(D/D_нач)²]")
    ax.set_title(f"рост в линейном режиме ({n_lin}³, амплитуда × 10⁻⁴)")
    ax.legend(fontsize=8)

    ax = axes[2]
    dm = d[key("dm", 1.0)].astype(float) - 1
    for name in KINDS:
        k, r = cross_correlation(d[key(name, 1.0)].astype(float) - 1, dm, length)
        ax.semilogx(k, r, color=COLORS[name], label=LABELS[name], ls=":" if name == "isothermal" else "-")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("k, h/Мпк")
    ax.set_ylabel("r(k) с тёмной материей")
    ax.set_title("насколько газ повторяет ТМ, z = 0")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / f"web3d_spectra_{n}.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(int(sys.argv[2]), float(sys.argv[3]))
    else:
        plot(int(sys.argv[2]))
