"""Тёмная материя частицами (particle-mesh) — опора для сравнения с газом в 3D.

Частицы стартуют из тех же начальных условий, что и газ (смещения Зельдовича
по тому же полю δ). Масса раскладывается на сетку и собирается обратно
ядром CIC (облако в ячейке). Пуассон решается так же, как у газа:
БПФ с функцией Грина семиточечного лапласиана, сила — центральные разности.
Шаги — кик–дрейф–кик в переменных u = a·v: du/dt = −∇φ, dx/dt = u/a².
"""
import numpy as np
import scipy.fft as fft


class ParticleMesh:
    def __init__(self, n, length, cosmo):
        self.n, self.length, self.cosmo = n, length, cosmo
        self.dx = length / n
        k = 2 * np.pi * np.fft.fftfreq(n, self.dx)
        kz = 2 * np.pi * np.fft.rfftfreq(n, self.dx)
        lap = sum((4 / self.dx**2) * np.sin(kc * self.dx / 2) ** 2
                  for kc in np.meshgrid(k, k, kz, indexing="ij", sparse=True))
        lap[0, 0, 0] = 1.0
        self._inv_lap = -1.0 / lap
        self._inv_lap[0, 0, 0] = 0.0

    def _cic(self, pos):
        """Восемь соседних ячеек каждой частицы: плоские индексы и веса."""
        n = self.n
        s = pos / self.dx - 0.5  # центры ячеек в (i + 1/2)·dx
        i0 = np.floor(s).astype(np.int64)
        f = s - i0
        out = []
        for ox in (0, 1):
            for oy in (0, 1):
                for oz in (0, 1):
                    off = np.array([ox, oy, oz])[:, None]
                    idx = (i0 + off) % n
                    w = np.prod(np.where(off == 1, f, 1 - f), axis=0)
                    out.append(((idx[0] * n + idx[1]) * n + idx[2], w))
        return out

    def density(self, pos, corners=None):
        """ρ/ρ̄ на сетке."""
        n = self.n
        corners = self._cic(pos) if corners is None else corners
        rho = np.zeros(n**3)
        for idx, w in corners:
            rho += np.bincount(idx, w, minlength=n**3)
        return rho.reshape(n, n, n) * n**3 / pos.shape[1]

    def accel(self, pos, a):
        """−∇φ в положениях частиц."""
        n = self.n
        corners = self._cic(pos)
        rho = self.density(pos, corners)
        src = self.cosmo.four_pi_g / a * (rho - 1)
        phi = fft.irfftn(fft.rfftn(src, workers=-1) * self._inv_lap, s=(n, n, n), workers=-1)
        acc = np.zeros_like(pos)
        for d in range(3):
            grad = ((np.roll(phi, -1, d) - np.roll(phi, 1, d)) / (2 * self.dx)).ravel()
            for idx, w in corners:
                acc[d] -= grad[idx] * w
        return acc

    def run(self, pos, u, a_start, a_out, dlna=0.01, callback=None):
        """Шагает от a_start через все a_out; в каждой точке вызывает callback(a, pos, u)."""
        c = self.cosmo
        a_grid = np.unique(np.concatenate([np.exp(np.arange(np.log(a_start), np.log(max(a_out)), dlna)),
                                           a_out]))
        t_grid = c.t_of_a(a_grid)
        acc = self.accel(pos, a_start)
        for i in range(len(a_grid) - 1):
            dt = t_grid[i + 1] - t_grid[i]
            a_mid = c.a_of_t(t_grid[i] + dt / 2)
            u += 0.5 * dt * acc
            pos = (pos + u * dt / a_mid**2) % self.length
            acc = self.accel(pos, a_grid[i + 1])
            u += 0.5 * dt * acc
            if a_grid[i + 1] in a_out and callback is not None:
                callback(a_grid[i + 1], pos, u)
        return pos, u


def pad_modes(delta_k, n, m):
    """rfftn-образ поля n³ → поле m³ с теми же модами (интерполяция Фурье), m ≥ n."""
    h = n // 2
    rows, src = np.r_[0:h, m - h:m], np.r_[0:h, n - h:n]
    out = np.zeros((m, m, m // 2 + 1), dtype=complex)
    out[np.ix_(rows, rows, np.arange(h + 1))] = delta_k[np.ix_(src, src, np.arange(h + 1))]
    return out * (m / n) ** 3


def zeldovich_particles(delta_k, n, length, cosmo, a, n_part=None):
    """Частицы в узлах решётки n_part³ (узел 0 — центр ячейки 0 сетки n³), смещённые по Зельдовичу: ψ_k = i·k/k²·δ_k, u = a·(aHf·ψ).

    При n_part > n моды поля те же, что у сетки n³: частиц больше, но начальные условия
    совпадают с газовыми.
    """
    offset = 0.5 * length / n  # узел 0 решётки — центр ячейки 0 сетки поля
    if n_part is not None and n_part != n:
        delta_k, n = pad_modes(delta_k, n, n_part), n_part
    k = 2 * np.pi * np.fft.fftfreq(n, length / n)
    kz = 2 * np.pi * np.fft.rfftfreq(n, length / n)
    kx, ky, kz = np.meshgrid(k, k, kz, indexing="ij", sparse=True)
    k2 = kx**2 + ky**2 + kz**2
    k2[0, 0, 0] = 1.0
    psi = np.array([fft.irfftn(1j * kc / k2 * delta_k, s=(n, n, n), workers=-1) for kc in (kx, ky, kz)])
    q = offset + np.arange(n) * length / n
    grid = np.array(np.meshgrid(q, q, q, indexing="ij"))
    pos = ((grid + psi) % length).reshape(3, -1)
    u = (a * a * cosmo.hubble(a) * cosmo.growth_rate(a) * psi).reshape(3, -1)
    return pos, u
