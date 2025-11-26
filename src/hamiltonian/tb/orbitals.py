from __future__ import print_function, division
from collections import OrderedDict
import sys
from .aux_functions import print_table

class Orbitals(object):
    orbital_sets=OrderedDict()
    @staticmethod
    def atoms_factory(labels):
        output=OrderedDict()
        for label in labels:
            base=''.join([i for i in label if not i.isdigit()])
            if base in output:
                continue
            if base in Orbitals.orbital_sets and isinstance(Orbitals.orbital_sets[base], Orbitals):
                atom1=Orbitals.orbital_sets[base]
            else:
                if base.lower()=='si':
                    atom1=SiliconSP3D5S()
                elif base.lower()=='h':
                    atom1=HydrogenS()
                elif base.lower()=='c':
                    atom1=CarbonPz()
                elif base.lower()=='cg':
                    atom1=CarbonPz()
                elif base.lower() == 'base':
                    atom1 = Base()
                else:
                    raise ValueError('There is no library entry for the atom '+label)
            output[atom1.title]=atom1
        return output
    def __init__(self, title):
        self.title=title
        self.orbitals=[]
        self.num_of_orbitals=0
        Orbitals.orbital_sets[self.title]=self
    def add_orbital(self, title, energy=0.0, principal=0, orbital=0, magnetic=0, spin=0):
        orbital={'title':title, 'energy':energy, 'n':principal, 'l':orbital, 'm':magnetic, 's':spin}
        self.orbitals.append(orbital)
        self.num_of_orbitals+=1
        Orbitals.orbital_sets[self.title]=self
    def set_orbital_energy(self, title, energy):
        """Update the onsite energy of an orbital identified by its title."""
        for orbital in self.orbitals:
            if orbital['title'] == title:
                orbital['energy'] = energy
                Orbitals.orbital_sets[self.title] = self
                return
        raise ValueError('Orbital %s not found in %s' % (title, self.title))
    @classmethod
    def edit_orbital(cls, atom_title, orbital_title, energy):
        """Convenience helper to tweak an atom's orbital energy at runtime."""
        base=''.join([i for i in atom_title if not i.isdigit()])
        if base in cls.orbital_sets and isinstance(cls.orbital_sets[base], Orbitals):
            cls.orbital_sets[base].set_orbital_energy(orbital_title, energy)
            return
        raise KeyError('Atom %s not found in orbital registry' % atom_title)
    def generate_info(self):
        return print_table(self.orbitals)

class SiliconSP3D5S(Orbitals):
    def __init__(self):
        super(SiliconSP3D5S, self).__init__('Si')
        self.add_orbital('s', energy=-2.0196, spin=0)
        self.add_orbital('c', energy=19.6748, principal=1, spin=0)
        self.add_orbital('px', energy=4.5448, orbital=1, magnetic=-1, spin=0)
        self.add_orbital('py', energy=4.5448, orbital=1, magnetic=1, spin=0)
        self.add_orbital('pz', energy=4.5448, orbital=1, magnetic=0, spin=0)
        self.add_orbital('dz2', energy=14.1836, orbital=2, magnetic=-1, spin=0)
        self.add_orbital('dxz', energy=14.1836, orbital=2, magnetic=-2, spin=0)
        self.add_orbital('dyz', energy=14.1836, orbital=2, magnetic=2, spin=0)
        self.add_orbital('dxy', energy=14.1836, orbital=2, magnetic=1, spin=0)
        self.add_orbital('dx2my2', energy=14.1836, orbital=2, magnetic=0, spin=0)

class HydrogenS(Orbitals):
    def __init__(self):
        super(HydrogenS, self).__init__('H')
        self.add_orbital('s', energy=0.9998)
        

class CarbonSP(Orbitals):
    def __init__(self):
        super(CarbonSP, self).__init__('C')
        # Empirical onsite energies for sp2-bonded carbon
        self.add_orbital('s', energy=-8.97, spin=0)
        self.add_orbital('px', energy=-3.34, orbital=1, magnetic=-1, spin=0)
        self.add_orbital('py', energy=-3.34, orbital=1, magnetic=1, spin=0)
        self.add_orbital('pz', energy=-3.34, orbital=1, magnetic=0, spin=0)

class CarbonPz(Orbitals):
    def __init__(self):
        super(CarbonPz, self).__init__('Cg')
        self.add_orbital('pz', energy=0.0, orbital=1, magnetic=0, spin=0)


class Base(Orbitals):
    def __init__(self):
        super(Base, self).__init__('base')
        self.add_orbital('s', energy=0)
