""" copied from nanonet
"""
from __future__ import annotations

class Orbitals:
    orbital_sets = {}

    def __init__(self, title: str):
        self.title = title
        self.orbitals = []  # list of dicts
        self.num_of_orbitals = 0
        Orbitals.orbital_sets[title] = self

    def add_orbital(self, title, energy=0.0, n=0, l=0, m=0, s=0):
        self.orbitals.append({'title': title, 'energy': energy, 'n': n, 'l': l, 'm': m, 's': s})
        self.num_of_orbitals += 1


class SiliconSP3D5S(Orbitals):
    """Silicon sp3d5s* basis (10 orbitals, spinless)"""
    def __init__(self):
        super().__init__('Si')
        self.add_orbital('s', energy=-2.0196, n=0, l=0, m=0, s=0)
        self.add_orbital('c', energy=19.6748, n=1, l=0, m=0, s=0)  # excited s* ("c")
        # In NanoNet px/py/pz have n=0
        self.add_orbital('px', energy=4.5448, n=0, l=1, m=-1, s=0)
        self.add_orbital('py', energy=4.5448, n=0, l=1, m=1, s=0)
        self.add_orbital('pz', energy=4.5448, n=0, l=1, m=0, s=0)
        # d orbitals also n=0
        self.add_orbital('dz2', energy=14.1836, n=0, l=2, m=-1, s=0)
        self.add_orbital('dxz', energy=14.1836, n=0, l=2, m=-2, s=0)
        self.add_orbital('dyz', energy=14.1836, n=0, l=2, m=2, s=0)
        self.add_orbital('dxy', energy=14.1836, n=0, l=2, m=1, s=0)
        self.add_orbital('dx2my2', energy=14.1836, n=0, l=2, m=0, s=0)


class HydrogenS(Orbitals):
    def __init__(self):
        super().__init__('H')
        self.add_orbital('s', energy=0.9998, n=0, l=0, m=0, s=0)
