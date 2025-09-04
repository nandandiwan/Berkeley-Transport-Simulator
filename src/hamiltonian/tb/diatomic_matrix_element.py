from __future__ import print_function, absolute_import, division
import sys, math
from .constants import *
from . import tb_params

def me_diatomic(bond, n, l_min, l_max, m, which_neighbour, overlap=False):
    label = n[0] + ORBITAL_QN[l_min] + n[1] + ORBITAL_QN[l_max] + '_' + M_QN[m]
    flag = 'OV_' if overlap else 'PARAMS_'
    try:
        if which_neighbour == 0:
            return getattr(sys.modules[tb_params.__name__], flag + bond)[label]
        elif which_neighbour == 100:
            return 0
        else:
            return getattr(sys.modules[tb_params.__name__], flag + bond + str(which_neighbour))[label]
    except KeyError:
        return 0

def d_me(N, l, m1, m2):
    if N == -1:
        N += sys.float_info.epsilon
    prefactor = ((0.5 * (1 + N)) ** l) * (((1 - N) / (1 + N)) ** (m1 * 0.5 - m2 * 0.5)) * \
                math.sqrt(math.factorial(l + m2) * math.factorial(l - m2) * math.factorial(l + m1) * math.factorial(l - m1))
    ans = 0
    for t in range(2 * l + 2):
        if l + m2 - t >= 0 and l - m1 - t >= 0 and t + m1 - m2 >= 0:
            if N == -1.0 and t == 0:
                ans += ((-1) ** t) / (math.factorial(l + m2 - t) * math.factorial(l - m1 - t) * math.factorial(t) * math.factorial(t + m1 - m2))
            else:
                ans += ((-1) ** t) * (((1 - N) / (1 + N)) ** t) / (math.factorial(l + m2 - t) * math.factorial(l - m1 - t) * math.factorial(t) * math.factorial(t + m1 - m2))
    return ans * prefactor

def tau(m):
    return 0 if m < 0 else 1

def a_coef(m, gamma):
    if m == 0:
        return 1.0 / math.sqrt(2)
    return ((-1) ** abs(m)) * (tau(m) * math.cos(abs(m) * gamma) - tau(-m) * math.sin(abs(m) * gamma))

def b_coef(m, gamma):
    return ((-1) ** abs(m)) * (tau(m) * math.sin(abs(m) * gamma) + tau(-m) * math.cos(abs(m) * gamma))

def s_me(N, l, m1, m2, gamma):
    return a_coef(m1, gamma) * (((-1) ** abs(m2)) * d_me(N, l, abs(m1), abs(m2)) + d_me(N, l, abs(m1), -abs(m2)))

def t_me(N, l, m1, m2, gamma):
    if m1 == 0:
        return 0
    return b_coef(m1, gamma) * (((-1) ** abs(m2)) * d_me(N, l, abs(m1), abs(m2)) - d_me(N, l, abs(m1), -abs(m2)))

def me(atom1, ll1, atom2, ll2, coords, which_neighbour=0, overlap=False):
    atoms = sorted([item.upper() for item in [atom1.title, atom2.title]])
    atoms = atoms[0] + '_' + atoms[1]
    n1 = atom1.orbitals[ll1]['n']; l1 = atom1.orbitals[ll1]['l']; m1 = atom1.orbitals[ll1]['m']; s1 = atom1.orbitals[ll1]['s']
    n2 = atom2.orbitals[ll2]['n']; l2 = atom2.orbitals[ll2]['l']; m2 = atom2.orbitals[ll2]['m']; s2 = atom2.orbitals[ll2]['s']
    if s1 == s2:
        L, M, N = coords[0], coords[1], coords[2]
        gamma = math.atan2(L, M)
        if l1 > l2:
            code = [n2, n1]
        elif l1 == l2:
            code = [min(n1, n2), max(n1, n2)]
        else:
            code = [n1, n2]
        for j, item in enumerate(code):
            code[j] = "" if item == 0 else str(item)
        l_min = min(l1, l2); l_max = max(l1, l2)
        prefactor = (-1) ** ((l1 - l2 + abs(l1 - l2)) * 0.5)
        ans = 2 * a_coef(m1, gamma) * a_coef(m2, gamma) * d_me(N, l1, abs(m1), 0) * d_me(N, l2, abs(m2), 0) * \
              me_diatomic(atoms, code, l_min, l_max, 0, which_neighbour, overlap=overlap)
        for m in range(1, l_min + 1):
            ans += (s_me(N, l1, m1, m, gamma) * s_me(N, l2, m2, m, gamma) + t_me(N, l1, m1, m, gamma) * t_me(N, l2, m2, m, gamma)) * \
                   me_diatomic(atoms, code, l_min, l_max, m, which_neighbour, overlap=overlap)
        return prefactor * ans
    else:
        return 0
