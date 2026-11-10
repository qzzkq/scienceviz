"""Точные решения задачи Римана для проверки схемы.

Идеальный газ — по Торо («Riemann Solvers and Numerical Methods for Fluid
Dynamics», гл. 4). Изотермический газ — та же конструкция, но вместо
давления неизвестна плотность в средней области.

Состояния передаются кортежами: (ρ, v, P) для идеального газа, (ρ, v) для
изотермического. Решение возвращается в точках ξ = (x − x0) / t.
"""
import numpy as np
from scipy.optimize import brentq


def _bracket(g, lo, hi):
    while g(lo) > 0:
        lo /= 10
    while g(hi) < 0:
        hi *= 10
    return lo, hi


# --- идеальный газ ---

def _f_ideal(p, rho_k, p_k, gamma):
    c_k = np.sqrt(gamma * p_k / rho_k)
    if p > p_k:  # ударная волна
        a_k = 2 / ((gamma + 1) * rho_k)
        b_k = (gamma - 1) / (gamma + 1) * p_k
        return (p - p_k) * np.sqrt(a_k / (p + b_k))
    return 2 * c_k / (gamma - 1) * ((p / p_k) ** ((gamma - 1) / (2 * gamma)) - 1)


def ideal_star(left, right, gamma):
    """Давление и скорость в средней области."""
    rl, vl, pl = left
    rr, vr, pr = right
    g = lambda p: _f_ideal(p, rl, pl, gamma) + _f_ideal(p, rr, pr, gamma) + vr - vl
    p_star = brentq(g, *_bracket(g, 1e-6 * min(pl, pr), max(pl, pr)), xtol=1e-15, rtol=1e-14)
    v_star = 0.5 * (vl + vr) + 0.5 * (_f_ideal(p_star, rr, pr, gamma) - _f_ideal(p_star, rl, pl, gamma))
    return p_star, v_star


def ideal_sample(left, right, gamma, xi):
    """Плотность, скорость и давление в точках ξ."""
    p_star, v_star = ideal_star(left, right, gamma)
    g1 = (gamma - 1) / (gamma + 1)
    out = np.empty((3, len(xi)))
    for i, s in enumerate(xi):
        # решение справа — зеркальное отражение решения слева
        if s <= v_star:
            (rk, vk, pk), sign = left, 1
        else:
            (rk, vk, pk), sign = right, -1
        ck = np.sqrt(gamma * pk / rk)
        # в отражённых координатах волна всегда левая
        s_, vk_, v_star_ = sign * s, sign * vk, sign * v_star
        if p_star > pk:
            shock = vk_ - ck * np.sqrt((gamma + 1) / (2 * gamma) * p_star / pk + (gamma - 1) / (2 * gamma))
            if s_ <= shock:
                w = (rk, vk_, pk)
            else:
                w = (rk * (p_star / pk + g1) / (g1 * p_star / pk + 1), v_star_, p_star)
        else:
            head = vk_ - ck
            tail = v_star_ - ck * (p_star / pk) ** ((gamma - 1) / (2 * gamma))
            if s_ <= head:
                w = (rk, vk_, pk)
            elif s_ >= tail:
                w = (rk * (p_star / pk) ** (1 / gamma), v_star_, p_star)
            else:
                v = 2 / (gamma + 1) * (ck + (gamma - 1) / 2 * vk_ + s_)
                c = 2 / (gamma + 1) * (ck + (gamma - 1) / 2 * (vk_ - s_))
                w = (rk * (c / ck) ** (2 / (gamma - 1)), v, pk * (c / ck) ** (2 * gamma / (gamma - 1)))
        out[:, i] = w[0], sign * w[1], w[2]
    return out


# --- изотермический газ ---

def _f_iso(rho, rho_k, cs):
    if rho > rho_k:  # ударная волна
        return cs * (rho - rho_k) / np.sqrt(rho * rho_k)
    return cs * np.log(rho / rho_k)


def isothermal_star(left, right, cs):
    """Плотность и скорость в средней области."""
    rl, vl = left
    rr, vr = right
    g = lambda r: _f_iso(r, rl, cs) + _f_iso(r, rr, cs) + vr - vl
    rho_star = brentq(g, *_bracket(g, min(rl, rr), max(rl, rr)), xtol=1e-15, rtol=1e-14)
    v_star = 0.5 * (vl + vr) + 0.5 * (_f_iso(rho_star, rr, cs) - _f_iso(rho_star, rl, cs))
    return rho_star, v_star


def isothermal_sample(left, right, cs, xi):
    """Плотность и скорость в точках ξ."""
    rho_star, v_star = isothermal_star(left, right, cs)
    out = np.empty((2, len(xi)))
    for i, s in enumerate(xi):
        if s <= v_star:
            (rk, vk), sign = left, 1
        else:
            (rk, vk), sign = right, -1
        s_, vk_, v_star_ = sign * s, sign * vk, sign * v_star
        if rho_star > rk:
            shock = vk_ - cs * np.sqrt(rho_star / rk)
            w = (rk, vk_) if s_ <= shock else (rho_star, v_star_)
        else:
            head, tail = vk_ - cs, v_star_ - cs
            if s_ <= head:
                w = (rk, vk_)
            elif s_ >= tail:
                w = (rho_star, v_star_)
            else:
                v = s_ + cs
                w = (rk * np.exp((vk_ - v) / cs), v)
        out[:, i] = w[0], sign * w[1]
    return out
