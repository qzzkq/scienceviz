"""1D-прототип схемы газа: конечные объёмы, HLL, сопутствующие координаты.

Уравнения, переменные и порядок шага описаны в equations.md. Состояние U
хранится по строкам: ρ, m = ρv, а у адиабатического газа ещё E (полная
энергия) и Ŝ = ρ·(P/ρ^γ)·a^{3(γ−1)} (энтропия для двойной энергии).

Без космологии (cosmo=None) a = 1, расширения и гравитации нет — так
считается ударная труба.
"""
from dataclasses import dataclass

import numpy as np

RHO, MOM, ENE, ENT = range(4)


@dataclass
class Gas:
    kind: str                  # "dust", "isothermal" или "adiabatic"
    cs: float = 0.0            # скорость звука изотермического газа
    gamma: float = 5 / 3
    dual_energy: bool = True   # в холодных потоках брать давление из энтропии
    eta: float = 0.01          # порог двойной энергии: доля тепловой энергии от max E среди соседей
    e_floor: float = 1e-12     # нижний порог удельной внутренней энергии без двойной энергии

    @property
    def nvar(self):
        return 4 if self.kind == "adiabatic" else 2

    def pressure(self, U, a, e_max=None):
        """Давление; e_max — максимум полной энергии по ячейке и её соседям (для двойной энергии)."""
        rho = U[RHO]
        if self.kind == "dust":
            return np.zeros_like(rho)
        if self.kind == "isothermal":
            return self.cs**2 * rho
        g = self.gamma
        p_e = (g - 1) * (U[ENE] - 0.5 * U[MOM] ** 2 / rho)
        if not self.dual_energy:
            return np.maximum(p_e, (g - 1) * rho * self.e_floor)
        p_s = np.maximum(U[ENT] * rho ** (g - 1) * a ** (-3 * (g - 1)), (g - 1) * rho * self.e_floor)
        e_max = U[ENE] if e_max is None else e_max
        return np.where(p_e > (g - 1) * self.eta * e_max, p_e, p_s)

    def sound(self, rho, p):
        if self.kind == "dust":
            return np.zeros_like(rho)
        if self.kind == "isothermal":
            return np.full_like(rho, self.cs)
        return np.sqrt(self.gamma * p / rho)

    def conserved(self, rho, v, p, a):
        U = [rho, rho * v]
        if self.kind == "adiabatic":
            g = self.gamma
            U += [p / (g - 1) + 0.5 * rho * v**2, p * rho ** (1 - g) * a ** (3 * (g - 1))]
        return np.array(U)

    def flux(self, rho, v, p):
        """Физический поток ρ, ρv и E без множителя 1/a; поток Ŝ считается отдельно, в hll()."""
        F = [rho * v, rho * v**2 + p]
        if self.kind == "adiabatic":
            e_tot = p / (self.gamma - 1) + 0.5 * rho * v**2
            F.append((e_tot + p) * v)
        return np.array(F)

    def sync(self, U, a, e_max=None):
        """Привести E и Ŝ к выбранному давлению (двойная энергия)."""
        if self.kind != "adiabatic":
            return
        g = self.gamma
        rho = U[RHO]
        p = self.pressure(U, a, e_max)
        U[ENE] = p / (g - 1) + 0.5 * U[MOM] ** 2 / rho
        U[ENT] = p * rho ** (1 - g) * a ** (3 * (g - 1))


def mc_slopes(w):
    """Наклоны с ограничителем MC для ячеек 1..n−2 массива w (по последней оси)."""
    dl = w[..., 1:-1] - w[..., :-2]
    dr = w[..., 2:] - w[..., 1:-1]
    s = np.sign(dl + dr) * np.minimum(np.minimum(2 * np.abs(dl), 2 * np.abs(dr)), 0.5 * np.abs(dl + dr))
    return np.where(dl * dr > 0, s, 0.0)


def hll(gas, wl, wr, a):
    """Поток через грани по левым и правым примитивным состояниям (ρ, v, P[, s = Ŝ/ρ]).

    ρ, ρv и E — по HLL. Энтропия переносится потоком массы: Ŝ-поток = F_ρ·s с той
    стороны, откуда течёт вещество. Так Ŝ остаётся положительной: если брать её
    из P и ρ на грани, в холодном газе за шаг может вытечь больше, чем было.
    """
    rl, vl, pl = wl[:3]
    rr, vr, pr = wr[:3]
    cl, cr = gas.sound(rl, pl), gas.sound(rr, pr)
    sl = np.minimum(np.minimum(vl - cl, vr - cr), 0.0)
    sr = np.maximum(np.maximum(vl + cl, vr + cr), 0.0)
    fl, fr = gas.flux(rl, vl, pl), gas.flux(rr, vr, pr)
    nh = len(fl)
    ul, ur = gas.conserved(rl, vl, pl, a)[:nh], gas.conserved(rr, vr, pr, a)[:nh]
    # при sl = sr = 0 (пыль в покое) поток и так нулевой
    width = np.where(sr - sl > 0, sr - sl, 1.0)
    f = (sr * fl - sl * fr + sl * sr * (ur - ul)) / width
    if gas.kind != "adiabatic":
        return f
    return np.concatenate([f, [f[0] * np.where(f[0] > 0, wl[3], wr[3])]])


class Sim1D:
    def __init__(self, gas, U, length, order=1, bc="periodic", cosmo=None, t=0.0,
                 cfl=0.4, max_dlna=0.01, rho_floor=1e-10):
        self.gas, self.U, self.length = gas, np.array(U, dtype=float), length
        self.n = self.U.shape[1]
        self.dx = length / self.n
        self.order, self.bc, self.cosmo = order, bc, cosmo
        self.t, self.cfl, self.max_dlna, self.rho_floor = t, cfl, max_dlna, rho_floor
        self.steps = 0
        self._grad_phi = None  # градиент потенциала с прошлого шага: кики плотность не меняют

    @property
    def a(self):
        return self._a(self.t)

    def _a(self, t):
        return 1.0 if self.cosmo is None else self.cosmo.a_of_t(t)

    def _pad(self, U, ng):
        if self.bc == "periodic":
            return np.concatenate([U[:, -ng:], U, U[:, :ng]], axis=1)
        return np.concatenate([np.repeat(U[:, :1], ng, axis=1), U, np.repeat(U[:, -1:], ng, axis=1)], axis=1)

    def _energy_max(self, U):
        """Максимум полной энергии по ячейке и двум соседям."""
        if self.gas.kind != "adiabatic":
            return None
        e = self._pad(U[ENE:ENE + 1], 1)[0]
        return np.maximum(np.maximum(e[:-2], e[1:-1]), e[2:])

    def primitives(self, U=None, a=None):
        U = self.U if U is None else U
        a = self.a if a is None else a
        rho = U[RHO]
        return rho, U[MOM] / rho, self.gas.pressure(U, a, self._energy_max(U))

    def rhs(self, U, a):
        """−(1/a)·(F_{i+1/2} − F_{i−1/2})/dx."""
        ng, n = self.order, self.n
        prims = list(self.primitives(U, a))
        if self.gas.kind == "adiabatic":
            prims.append(U[ENT] / U[RHO])
        w = self._pad(np.array(prims), ng)
        if self.order == 1:
            wl, wr = w[:, ng - 1:ng + n], w[:, ng:ng + n + 1]
        else:
            s = mc_slopes(w)  # s[:, k − 1] — наклон в ячейке k
            wl = w[:, ng - 1:ng + n] + 0.5 * s[:, ng - 2:ng - 1 + n]
            wr = w[:, ng:ng + n + 1] - 0.5 * s[:, ng - 1:ng + n]
        f = hll(self.gas, wl, wr, a) / a
        return -(f[:, 1:] - f[:, :-1]) / self.dx

    def timestep(self):
        rho, v, p = self.primitives()
        a = self.a
        speed = np.max(np.abs(v) + self.gas.sound(rho, p))
        dt = self.cfl * a * self.dx / speed if speed > 0 else np.inf
        if self.cosmo is not None:
            dt = min(dt, self.max_dlna / self.cosmo.hubble(a))
        return dt

    def grad_phi(self, rho, a):
        """∂φ/∂x: Пуассон через БПФ с функцией Грина дискретного лапласиана, центральные разности."""
        k = 2 * np.pi * np.fft.fftfreq(self.n, self.dx)
        lap = -(2 - 2 * np.cos(k * self.dx)) / self.dx**2
        lap[0] = 1.0
        src_k = np.fft.fft(self.cosmo.four_pi_g / a * (rho - rho.mean()))
        phi_k = src_k / lap
        phi_k[0] = 0.0
        phi = np.fft.ifft(phi_k).real
        return (np.roll(phi, -1) - np.roll(phi, 1)) / (2 * self.dx)

    def kick(self, grad_phi, dt, a_start, a_end, a_force, force_first):
        """Полшага источников: гравитация, хаббловское трение, остывание при расширении.

        Сила берётся в момент a_force. В первом кике шага её прирост ещё проходит
        трение, во втором — нет: так суммарно выходит формула трапеций, второй порядок.
        """
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
            U[ENE] = p / (g - 1) + 0.5 * U[MOM] ** 2 / rho
            U[ENT] = p * rho ** (1 - g) * a_end ** (3 * (g - 1))

    def hydro(self, dt, a):
        U = self.U
        u1 = U + dt * self.rhs(U, a)
        if self.order == 2:  # SSP-RK2
            u1[RHO] = np.maximum(u1[RHO], self.rho_floor)
            u1 = 0.5 * U + 0.5 * (u1 + dt * self.rhs(u1, a))
        u1[RHO] = np.maximum(u1[RHO], self.rho_floor)
        self.U = u1

    def step(self, dt):
        if self.cosmo is None:
            self.hydro(dt, 1.0)
            self.gas.sync(self.U, 1.0, self._energy_max(self.U))
        else:
            t0, tm, t1 = self.t, self.t + dt / 2, self.t + dt
            a0, am, a1 = self._a(t0), self._a(tm), self._a(t1)
            if self._grad_phi is None:
                self._grad_phi = self.grad_phi(self.U[RHO], a0)
            self.kick(self._grad_phi, dt / 2, a0, am, a0, force_first=True)
            self.hydro(dt, am)
            # остывание за вторую половину шага применит второй кик
            self.gas.sync(self.U, am, self._energy_max(self.U))
            self._grad_phi = self.grad_phi(self.U[RHO], a1)
            self.kick(self._grad_phi, dt / 2, am, a1, a1, force_first=False)
        self.t += dt
        self.steps += 1

    def run_to_time(self, t_end):
        while self.t < t_end * (1 - 1e-12):
            self.step(min(self.timestep(), t_end - self.t))

    def run_to_a(self, a_end):
        self.run_to_time(self.cosmo.t_of_a(a_end))

    def mass(self):
        return self.U[RHO].sum() * self.dx
