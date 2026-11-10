"""3D-прототип ядра: та же схема, что в hydro1d.py, на периодической сетке.

Без MPI и гало: соседи берутся через np.roll по периодическим границам.
Формулы и порядок шага — equations.md, разделы 4–7. Состояние U хранится по
строкам: ρ, ρv_x, ρv_y, ρv_z, а у адиабатического газа ещё E и Ŝ.
"""
import numpy as np
import scipy.fft as fft

from hydro1d import Gas

RHO, ENE, ENT = 0, 4, 5
MOM = slice(1, 4)


def mc_slope(w, axis):
    """Наклон с ограничителем MC вдоль оси axis (периодически)."""
    dl = w - np.roll(w, 1, axis)
    dr = np.roll(w, -1, axis) - w
    s = np.sign(dl + dr) * np.minimum(np.minimum(2 * np.abs(dl), 2 * np.abs(dr)), 0.5 * np.abs(dl + dr))
    return np.where(dl * dr > 0, s, 0.0)


class Gas3D:
    """Уравнение состояния и потоки в 3D; параметры берутся из 1D-класса Gas."""

    def __init__(self, gas: Gas):
        self.gas = gas
        self.kind, self.gamma = gas.kind, gas.gamma
        self.nvar = 6 if gas.kind == "adiabatic" else 4

    def pressure(self, U, a):
        g, rho = self.gas, U[RHO]
        if g.kind == "dust":
            return np.zeros_like(rho)
        if g.kind == "isothermal":
            return g.cs**2 * rho
        gm = g.gamma
        p_e = (gm - 1) * (U[ENE] - 0.5 * (U[MOM] ** 2).sum(axis=0) / rho)
        if not g.dual_energy:
            return np.maximum(p_e, (gm - 1) * rho * g.e_floor)
        p_s = np.maximum(U[ENT] * rho ** (gm - 1) * a ** (-3 * (gm - 1)), (gm - 1) * rho * g.e_floor)
        e_max = U[ENE].copy()
        for axis in range(3):
            np.maximum(e_max, np.roll(U[ENE], 1, axis), out=e_max)
            np.maximum(e_max, np.roll(U[ENE], -1, axis), out=e_max)
        return np.where(p_e > (gm - 1) * g.eta * e_max, p_e, p_s)

    def sound(self, rho, p):
        if self.kind == "dust":
            return np.zeros_like(rho)
        if self.kind == "isothermal":
            return np.full_like(rho, self.gas.cs)
        return np.sqrt(self.gamma * p / rho)

    def conserved(self, rho, v, p, a):
        out = [rho, rho * v[0], rho * v[1], rho * v[2]]
        if self.kind == "adiabatic":
            g = self.gamma
            out += [p / (g - 1) + 0.5 * rho * (v**2).sum(axis=0), p * rho ** (1 - g) * a ** (3 * (g - 1))]
        return np.array(out)

    def flux(self, rho, v, p, d):
        """Физический поток ρ, ρv и E через грань, нормальную к оси d, без множителя 1/a."""
        mass = rho * v[d]
        out = [mass, mass * v[0], mass * v[1], mass * v[2]]
        out[1 + d] = out[1 + d] + p
        if self.kind == "adiabatic":
            e_tot = p / (self.gamma - 1) + 0.5 * rho * (v**2).sum(axis=0)
            out.append((e_tot + p) * v[d])
        return np.array(out)

    def hll(self, wl, wr, d, a):
        """Как hydro1d.hll: ρ, ρv, E по HLL, энтропия — потоком массы с наветренной стороны."""
        rl, vl, pl = wl[0], wl[1:4], wl[4]
        rr, vr, pr = wr[0], wr[1:4], wr[4]
        cl, cr = self.sound(rl, pl), self.sound(rr, pr)
        sl = np.minimum(np.minimum(vl[d] - cl, vr[d] - cr), 0.0)
        sr = np.maximum(np.maximum(vl[d] + cl, vr[d] + cr), 0.0)
        width = np.where(sr - sl > 0, sr - sl, 1.0)
        fl, fr = self.flux(rl, vl, pl, d), self.flux(rr, vr, pr, d)
        nh = len(fl)
        ul, ur = self.conserved(rl, vl, pl, a)[:nh], self.conserved(rr, vr, pr, a)[:nh]
        f = (sr * fl - sl * fr + sl * sr * (ur - ul)) / width
        if self.kind != "adiabatic":
            return f
        return np.concatenate([f, [f[0] * np.where(f[0] > 0, wl[5], wr[5])]])

    def sync(self, U, a):
        if self.kind != "adiabatic":
            return
        g = self.gamma
        rho = U[RHO]
        p = self.pressure(U, a)
        U[ENE] = p / (g - 1) + 0.5 * (U[MOM] ** 2).sum(axis=0) / rho
        U[ENT] = p * rho ** (1 - g) * a ** (3 * (g - 1))


class Sim3D:
    def __init__(self, gas, U, dx, cosmo, t, order=2, cfl=0.4, max_dlna=0.01, rho_floor=1e-10):
        self.gas = Gas3D(gas)
        self.U = np.array(U, dtype=float)
        self.shape = self.U.shape[1:]
        self.dx, self.cosmo, self.t, self.order = dx, cosmo, t, order
        self.cfl, self.max_dlna, self.rho_floor = cfl, max_dlna, rho_floor
        self.steps = 0
        self._grad_phi = None
        # собственные числа семиточечного лапласиана для rfftn-раскладки
        ks = [2 * np.pi * np.fft.fftfreq(n, dx) for n in self.shape[:2]] + \
             [2 * np.pi * np.fft.rfftfreq(self.shape[2], dx)]
        lap = sum((4 / dx**2) * np.sin(k * dx / 2) ** 2 for k in np.meshgrid(*ks, indexing="ij", sparse=True))
        lap[0, 0, 0] = 1.0
        self._inv_lap = -1.0 / lap
        self._inv_lap[0, 0, 0] = 0.0

    @property
    def a(self):
        return self.cosmo.a_of_t(self.t)

    def primitives(self, U=None, a=None):
        U = self.U if U is None else U
        a = self.a if a is None else a
        rho = U[RHO]
        return rho, U[MOM] / rho, self.gas.pressure(U, a)

    def rhs(self, U, a):
        rho, v, p = self.primitives(U, a)
        rows = [rho[None], v, p[None]]
        if self.gas.kind == "adiabatic":
            rows.append((U[ENT] / rho)[None])
        w = np.concatenate(rows)
        out = np.zeros_like(U)
        for d in range(3):
            if self.shape[d] == 1:
                continue
            axis = d + 1
            if self.order == 1:
                wl, wr = w, np.roll(w, -1, axis)
            else:
                s = mc_slope(w, axis)
                wl, wr = w + 0.5 * s, np.roll(w - 0.5 * s, -1, axis)
            f = self.gas.hll(wl, wr, d, a)  # поток через грань i+1/2
            out -= f - np.roll(f, 1, axis)
        return out / (a * self.dx)

    def timestep(self):
        rho, v, p = self.primitives()
        c = self.gas.sound(rho, p)
        active = [d for d in range(3) if self.shape[d] > 1]
        speed = np.max(sum(np.abs(v[d]) + c for d in active))
        a = self.a
        dt = self.cfl * a * self.dx / speed if speed > 0 else np.inf
        return min(dt, self.max_dlna / self.cosmo.hubble(a))

    def grad_phi(self, rho, a):
        src = self.cosmo.four_pi_g / a * (rho - rho.mean())
        phi = fft.irfftn(fft.rfftn(src, workers=-1) * self._inv_lap, s=self.shape, workers=-1)
        return np.array([(np.roll(phi, -1, d) - np.roll(phi, 1, d)) / (2 * self.dx)
                         if self.shape[d] > 1 else np.zeros_like(phi) for d in range(3)])

    def kick(self, grad_phi, dt, a_start, a_end, a_force, force_first):
        U, gas = self.U, self.gas
        rho = U[RHO]
        if gas.kind == "adiabatic":
            g = gas.gamma
            e = gas.pressure(U, a_start) / ((g - 1) * rho)
        force = dt * rho * grad_phi / a_force
        if force_first:
            U[MOM] = (U[MOM] - force) * (a_start / a_end)
        else:
            U[MOM] = U[MOM] * (a_start / a_end) - force
        if gas.kind == "adiabatic":
            p = (g - 1) * rho * e * (a_start / a_end) ** (3 * (g - 1))
            U[ENE] = p / (g - 1) + 0.5 * (U[MOM] ** 2).sum(axis=0) / rho
            U[ENT] = p * rho ** (1 - g) * a_end ** (3 * (g - 1))

    def hydro(self, dt, a):
        U = self.U
        u1 = U + dt * self.rhs(U, a)
        if self.order == 2:  # SSP-RK2
            np.maximum(u1[RHO], self.rho_floor, out=u1[RHO])
            u1 = 0.5 * U + 0.5 * (u1 + dt * self.rhs(u1, a))
        np.maximum(u1[RHO], self.rho_floor, out=u1[RHO])
        self.U = u1

    def step(self, dt):
        c = self.cosmo
        t0, tm, t1 = self.t, self.t + dt / 2, self.t + dt
        a0, am, a1 = c.a_of_t(t0), c.a_of_t(tm), c.a_of_t(t1)
        if self._grad_phi is None:
            self._grad_phi = self.grad_phi(self.U[RHO], a0)
        self.kick(self._grad_phi, dt / 2, a0, am, a0, force_first=True)
        self.hydro(dt, am)
        self.gas.sync(self.U, am)
        self._grad_phi = self.grad_phi(self.U[RHO], a1)
        self.kick(self._grad_phi, dt / 2, am, a1, a1, force_first=False)
        self.t = t1
        self.steps += 1

    def run_to_a(self, a_end):
        t_end = self.cosmo.t_of_a(a_end)
        while self.t < t_end * (1 - 1e-12):
            self.step(min(self.timestep(), t_end - self.t))

    def mass(self):
        return self.U[RHO].sum() * self.dx**3
