"""Фон плоской ΛCDM в единицах кода.

Единицы кода (см. equations.md, раздел 2): длина — Мпк/h, время — 1/H0,
скорость — 100 км/с, плотность — средняя сопутствующая плотность вещества.
В этих единицах H0 = 1, ρ̄ = 1 и 4πGρ̄ = 1.5·Ωm.
"""
import numpy as np
from scipy.integrate import quad

K_BOLTZMANN = 1.380649e-23   # Дж/К
M_PROTON = 1.67262192e-27    # кг
VELOCITY_UNIT = 1e5          # 100 км/с в м/с
MU = 0.6                     # средняя масса частицы в массах протона (ионизованный газ)


class Cosmology:
    """Плоская ΛCDM без излучения. По умолчанию — параметры Quijote."""

    def __init__(self, omega_m=0.3175, h=0.6711, omega_b=0.049, n_s=0.9624, sigma8=0.834):
        self.omega_m = omega_m
        self.omega_l = 1.0 - omega_m
        self.h = h
        self.omega_b = omega_b
        self.n_s = n_s
        self.sigma8 = sigma8

    @property
    def four_pi_g(self):
        """4πGρ̄ в единицах кода."""
        return 1.5 * self.omega_m

    def hubble(self, a):
        """H(a) / H0."""
        return np.sqrt(self.omega_m / a**3 + self.omega_l)

    def a_of_t(self, t):
        if self.omega_l == 0:
            return (1.5 * t) ** (2 / 3)
        sl = np.sqrt(self.omega_l)
        return (self.omega_m / self.omega_l) ** (1 / 3) * np.sinh(1.5 * sl * t) ** (2 / 3)

    def t_of_a(self, a):
        if self.omega_l == 0:
            return 2 / 3 * a**1.5
        sl = np.sqrt(self.omega_l)
        return 2 / (3 * sl) * np.arcsinh(np.sqrt(self.omega_l / self.omega_m) * a**1.5)

    def growth(self, a):
        """Растущая мода D(a), нормированная на D → a при a → 0 (формула Хита)."""
        # (a'·H)^-3, переписанное без особенности в нуле
        integrand = lambda x: x**1.5 / (self.omega_m + self.omega_l * x**3) ** 1.5
        a_arr = np.atleast_1d(np.asarray(a, dtype=float))
        d = np.array([2.5 * self.omega_m * self.hubble(ai) * quad(integrand, 0, ai, epsabs=0, epsrel=1e-12)[0]
                      for ai in a_arr])
        return d if np.ndim(a) else d[0]

    def growth_rate(self, a):
        """f = d ln D / d ln a."""
        e2 = self.hubble(a) ** 2
        return self.omega_m / (a**2 * e2) * (2.5 / self.growth(a) - 1.5 / a)


def temperature(e):
    """Температура в К по удельной внутренней энергии e в единицах кода (γ = 5/3)."""
    return (2 / 3) * MU * M_PROTON * e * VELOCITY_UNIT**2 / K_BOLTZMANN


def internal_energy(T):
    """Обратная к temperature()."""
    return T * K_BOLTZMANN / ((2 / 3) * MU * M_PROTON * VELOCITY_UNIT**2)
