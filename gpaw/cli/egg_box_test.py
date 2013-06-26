import numpy as np

from ase.atoms import Atoms
from ase.cli.run import Runner


class EggBoxTest(Runner):
    def build(self, name):
        return Atoms(name, pbc=True)

    def calculate(self, atoms, name):
        opts = self.opts
        
        atoms.calc.set(smearing={'type': 'Fermi-Dirac'})

        data = {'emax': [], 'fmax': [], 'energies': [], 'forces': [],
                'grid_spacings': []}
        for h in [0.18, 0.2]:
            L = 4.0  # approximate cell size
            L = round(L / h / 4) * 4 * h
            atoms.cell = [L, L, L]
            atoms.calc.set(h=h)
            energies = []
            forces = []
            for i in range(25):
                x = h * i / 48
                atoms.positions[0] = x
                e = atoms.calc.get_potential_energy(atoms,
                                                    force_consistent=True)
                energies.append(e)
                forces.append(atoms.get_forces()[0, 0])
            emax = np.ptp(energies)
            fmax = np.abs(forces).max() * 3**0.5
            self.log('%s %.3f Ang %.6f eV %.6f eV/Ang' % (name, h, emax, fmax))
            data['emax'].append(emax)
            data['fmax'].append(fmax)
            data['energies'].append(energies)
            data['forces'].append(forces)
            data['grid_spacings'].append(h)
        return data
