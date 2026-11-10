"""Быстрые проверки прототипа.

Запуск: python3 test_prototype.py (или python3 -m pytest, если он стоит).
"""
import numpy as np

from cosmology import Cosmology, internal_energy, temperature
from hydro1d import Gas, Sim1D
from hydro3d import Gas3D, Sim3D
from ics import LinearPower, initial_conditions, measure_power
from pm3d import ParticleMesh
from riemann import ideal_star, ideal_sample, isothermal_sample, isothermal_star
from shocktube import l1_error
from zeldovich import A_INIT, A_PRE, T_INIT, Pancake, make_gas, run_dark_matter, run_gas


def test_sod_exact_matches_toro():
    p_star, v_star = ideal_star((1.0, 0.0, 1.0), (0.125, 0.0, 0.1), 1.4)
    assert abs(p_star - 0.30313) < 1e-5 and abs(v_star - 0.92745) < 1e-5
    rho_left, rho_right = ideal_sample((1.0, 0.0, 1.0), (0.125, 0.0, 0.1), 1.4, np.array([0.5, 1.5]))[0]
    assert abs(rho_left - 0.42632) < 1e-5 and abs(rho_right - 0.26557) < 1e-5


def test_isothermal_exact_satisfies_jump_conditions():
    left, right, cs = (1.0, 0.0), (0.1, 0.0), 1.0
    rho_star, v_star = isothermal_star(left, right, cs)
    shock = right[1] + cs * np.sqrt(rho_star / right[0])
    # масса и импульс через правую ударную волну в её системе отсчёта
    mass_in, mass_out = right[0] * (right[1] - shock), rho_star * (v_star - shock)
    mom_in = right[0] * (right[1] - shock) ** 2 + cs**2 * right[0]
    mom_out = rho_star * (v_star - shock) ** 2 + cs**2 * rho_star
    assert abs(mass_in - mass_out) < 1e-12 and abs(mom_in - mom_out) < 1e-12
    rho, v = isothermal_sample(left, right, cs, np.array([-2.0, 0.5 * (shock + v_star), 3.0]))
    assert np.allclose(rho, [1.0, rho_star, 0.1]) and np.allclose(v, [0.0, v_star, 0.0])


def test_shock_tubes_converge():
    for test in ("sod", "iso"):
        for order, limit in ((1, 1.2e-2), (2, 3e-3)):
            coarse, fine = l1_error(test, 100, order), l1_error(test, 400, order)
            assert fine < limit and fine < 0.6 * coarse, (test, order, coarse, fine)


def test_background_cosmology():
    c = Cosmology()
    assert abs(c.growth(1e-3) / 1e-3 - 1) < 1e-3
    assert abs(c.a_of_t(c.t_of_a(0.37)) - 0.37) < 1e-12
    a = np.linspace(1e-4, 1, 200001)
    t_numeric = np.trapezoid(1 / (a * c.hubble(a)), a)
    assert abs(t_numeric - c.t_of_a(1.0)) < 1e-5
    eds = Cosmology(omega_m=1.0)
    assert abs(eds.growth(0.5) - 0.5) < 1e-10 and abs(eds.growth_rate(0.5) - 1) < 1e-10


def test_pancake_before_caustic():
    c = Cosmology()
    pancake = Pancake(c)
    n = 128
    exact, _ = pancake.cell_averages(n, A_PRE)
    for kind in ("dust", "isothermal", "adiabatic"):
        out, sim = run_gas(pancake, make_gas(kind, c, n), n, order=2, a_out=(A_PRE,))
        err = np.abs(out[A_PRE][0] - exact).sum() / exact.sum()
        assert err < 2e-3, (kind, err)
        assert abs(sim.mass() / pancake.length - 1) < 1e-13


def test_dual_energy_keeps_cold_gas_cold_and_heats_shocks():
    c = Cosmology()
    pancake = Pancake(c)
    n = 128
    x = -pancake.length / 2 + (np.arange(n) + 0.5) * pancake.length / n
    out, _ = run_gas(pancake, Gas("adiabatic"), n, order=2, a_out=(A_PRE, 1.0))
    for a in (A_PRE, 1.0):
        rho, _, p, _ = out[a]
        t = temperature(p / (2 / 3 * rho))
        t_adiabatic = T_INIT * rho ** (2 / 3) * (A_INIT / a) ** 2
        cold = np.abs(x) > 8
        assert np.abs(t[cold] / t_adiabatic[cold] - 1).max() < 0.1, a
    assert t.max() > 1e6


def test_dark_matter_sheets_follow_zeldovich():
    c = Cosmology()
    pancake = Pancake(c)
    q, x, _ = run_dark_matter(pancake, n_part=4096, a_out=(A_PRE,))[A_PRE]
    assert np.abs(x - pancake.x_of_q(q, A_PRE)).max() < 1e-4


def test_initial_conditions_have_requested_spectrum():
    c = Cosmology()
    power = LinearPower(c)
    assert abs(power.sigma(8.0) - c.sigma8) < 1e-6
    rho, vel = initial_conditions(64, 256.0, c, 0.02, seed=1)
    assert abs(rho.mean() - 1) < 1e-12
    k, p, p_exp, count = measure_power(rho - 1, 256.0, power, 0.02)
    good = count >= 100
    assert abs(np.mean(p[good] / p_exp[good]) - 1) < 0.05
    # одно зерно — одно поле
    rho2, _ = initial_conditions(64, 256.0, c, 0.02, seed=1)
    assert np.array_equal(rho, rho2)


def test_3d_reproduces_1d_along_every_axis():
    """Блин вдоль x, y и z в 3D-коде совпадает с 1D-кодом до округления, включая ударные волны."""
    c = Cosmology()
    pancake = Pancake(c)
    n = 32
    rho, mom = pancake.cell_averages(n, A_INIT)
    v = mom / rho
    gas = make_gas("adiabatic", c, n)
    p = (gas.gamma - 1) * rho * internal_energy(T_INIT)
    t0 = c.t_of_a(A_INIT)
    sim1 = Sim1D(gas, gas.conserved(rho, v, p, A_INIT), pancake.length, order=2, cosmo=c, t=t0)
    sim1.run_to_a(0.5)
    for axis in range(3):
        shape = [1, 1, 1]
        shape[axis] = n
        vel = np.zeros([3] + shape)
        vel[axis] = v.reshape(shape)
        U = Gas3D(gas).conserved(rho.reshape(shape), vel, p.reshape(shape), A_INIT)
        sim3 = Sim3D(gas, U, pancake.length / n, c, t0)
        sim3.run_to_a(0.5)
        assert np.abs(sim3.U[0].ravel() - sim1.U[0]).max() < 1e-9, axis


def test_particle_mesh_follows_zeldovich():
    """Тёмная материя частицами до схлопывания отстаёт от точного решения меньше чем на треть ячейки."""
    c = Cosmology()
    pancake = Pancake(c)
    n, length = 16, pancake.length
    q1 = -length / 2 + (np.arange(n) + 0.5) * length / n
    qx, qy, qz = [g.ravel() for g in np.meshgrid(q1, q1, q1, indexing="ij")]
    pos = np.array([pancake.x_of_q(qx, A_INIT), qy, qz]) % length
    u = np.zeros_like(pos)
    u[0] = A_INIT * pancake.velocity(qx, A_INIT)
    pos, _ = ParticleMesh(n, length, c).run(pos, u, A_INIT, (A_PRE,))
    err = (pos[0] - pancake.x_of_q(qx, A_PRE) + length / 2) % length - length / 2
    assert np.abs(err).max() < length / n / 3


def test_3d_linear_growth_matches_theory():
    """Малое возмущение растёт как D(a) и в газе, и в тёмной материи (на больших масштабах до 1%)."""
    import web3d
    from checks import growth_ratio

    c = Cosmology()
    n, length = 32, 64.0  # на 16³ даже первый слой k — уже 0,16 k_Найквиста, и схема теряет там 2–5%
    delta_k, delta, vel = web3d.initial_fields(n, length, c, scale=1e-8)
    out = {}
    web3d.run_gas("dust", delta, vel, n, length, c, out, dtype=np.float64)
    web3d.run_dark_matter(delta_k, n, length, c, out, dtype=np.float64)
    for start, name in ((delta, "dust"), (out["dm_init"] - 1, "dm")):
        k, ratio, _ = growth_ratio(start, web3d.A_INIT, out[web3d.key(name, 0.25)] - 1, 0.25, length, c)
        large = k < 0.1 * np.pi * n / length
        assert np.abs(ratio[large] - 1).max() < 0.01, (name, ratio[large])
    assert out["dust_mass_drift"].max() < 1e-12


if __name__ == "__main__":
    tests = [(name, f) for name, f in sorted(globals().items()) if name.startswith("test_")]
    for name, f in tests:
        f()
        print(f"ok  {name}")
    print(f"пройдено {len(tests)}")
