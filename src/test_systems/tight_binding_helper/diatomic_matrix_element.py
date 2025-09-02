"""Full diatomic Slater-Koster matrix element (ported from NanoNet logic).

This replaces the earlier simplified version to achieve numerical parity
with the reference implementation. Supports l up to d and sigma/pi/delta.
"""
from __future__ import annotations
import math, sys
from . import tb_params
from .constants import ORBITAL_QN, M_QN

def _me_diatomic(bond, ncode, l_min, l_max, m, which_neighbour, overlap=False):
    flag = 'OV_' if overlap else 'PARAMS_'
    label = ncode[0] + ORBITAL_QN[l_min] + ncode[1] + ORBITAL_QN[l_max] + '_' + M_QN[m]
    try:
        if which_neighbour == 0:
            return getattr(tb_params, flag + bond)[label]
        elif which_neighbour == 100:
            return 0
        else:
            return getattr(tb_params, flag + bond + str(which_neighbour))[label]
    except KeyError:
        return 0

def _d_me(N, l, m1, m2):
    if N == -1:
        N += sys.float_info.epsilon
    prefactor = ((0.5 * (1 + N)) ** l) * (((1 - N) / (1 + N)) ** (0.5 * (m1 - m2))) * \
                math.sqrt(math.factorial(l + m2) * math.factorial(l - m2) *
                          math.factorial(l + m1) * math.factorial(l - m1))
    acc = 0.0
    for t in range(2 * l + 2):
        if l + m2 - t >= 0 and l - m1 - t >= 0 and t + m1 - m2 >= 0:
            term = ((-1) ** t) * (((1 - N) / (1 + N)) ** t)
            denom = math.factorial(l + m2 - t) * math.factorial(l - m1 - t) * math.factorial(t) * math.factorial(t + m1 - m2)
            acc += term / denom
    return acc * prefactor

def _tau(m): return 0 if m < 0 else 1
def _a(m, gamma):
    if m == 0: return 1.0 / math.sqrt(2)
    return ((-1) ** abs(m)) * (_tau(m) * math.cos(abs(m) * gamma) - _tau(-m) * math.sin(abs(m) * gamma))
def _b(m, gamma):
    return ((-1) ** abs(m)) * (_tau(m) * math.sin(abs(m) * gamma) + _tau(-m) * math.cos(abs(m) * gamma))
def _s(N, l, m1, m2, gamma):
    return _a(m1, gamma) * (((-1) ** abs(m2)) * _d_me(N, l, abs(m1), abs(m2)) + _d_me(N, l, abs(m1), -abs(m2)))
def _t(N, l, m1, m2, gamma):
    if m1 == 0: return 0
    return _b(m1, gamma) * (((-1) ** abs(m2)) * _d_me(N, l, abs(m1), abs(m2)) - _d_me(N, l, abs(m1), -abs(m2)))

def me(atom1, l1, atom2, l2, direction, which_neighbour=0, overlap=False):
    if overlap:
        return 0.0
    # species order for parameter key
    species = sorted([atom1.title.upper(), atom2.title.upper()])
    bond = species[0] + '_' + species[1]
    o1 = atom1.orbitals[l1]; o2 = atom2.orbitals[l2]
    if o1['s'] != o2['s']:
        return 0.0
    n1, l1q, m1 = o1['n'], o1['l'], o1['m']
    n2, l2q, m2 = o2['n'], o2['l'], o2['m']
    L, M, N = direction
    gamma = math.atan2(L, M)
    if l1q > l2q: code = [str(n2) if n2 else '', str(n1) if n1 else '']
    elif l1q == l2q: code = [str(min(n1, n2)) if min(n1, n2) else '', str(max(n1, n2)) if max(n1, n2) else '']
    else: code = [str(n1) if n1 else '', str(n2) if n2 else '']
    l_min = min(l1q, l2q); l_max = max(l1q, l2q)
    pref = (-1) ** ((l1q - l2q + abs(l1q - l2q)) * 0.5)
    ans = 2 * _a(m1, gamma) * _a(m2, gamma) * _d_me(N, l1q, abs(m1), 0) * _d_me(N, l2q, abs(m2), 0) * \
          _me_diatomic(bond, code, l_min, l_max, 0, which_neighbour, overlap=overlap)
    for m in range(1, l_min + 1):
        ans += (_s(N, l1q, m1, m, gamma) * _s(N, l2q, m2, m, gamma) +
                _t(N, l1q, m1, m, gamma) * _t(N, l2q, m2, m, gamma)) * \
               _me_diatomic(bond, code, l_min, l_max, m, which_neighbour, overlap=overlap)
    return pref * ans
