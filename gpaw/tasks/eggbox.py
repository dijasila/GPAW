import optparse

import numpy as np
from ase.atoms import Atoms

from gpaw import FermiDirac
from gpaw.tasks.fdconv import ConvergenceTestTask


class EggboxTestTask(ConvergenceTestTask):
    taskname = 'eggbox'

    def __init__(self, **kwargs):
        """Calculate size of eggbox error.

        A single atom is translated from (0, 0, 0) to (h / 2, 0, 0) in
        25 steps in order to measure the eggbox error."""

        ConvergenceTestTask.__init__(self, **kwargs)

    def calculate(self, name, atoms):
        atoms.calc.set(occupations=FermiDirac(0.1),
                       kpts=[1, 1, 1])
        data = {}
        for g in self.grid_points:
            atoms.calc.set(gpts=(g, g, g))
            energies = []
            forces = []
            for i in range(25):
                x = self.L / g * i / 48
                atoms.positions[0] = x
                e = atoms.calc.get_potential_energy(atoms,
                                                    force_consistent=True)
                energies.append(e)
                forces.append(atoms.get_forces()[0,0])
            data[g] = (energies, forces)

        return data

    def analyse(self):
        for name, data in self.data.items():
            for g in self.grid_points:
                de = data[str(g)][0].ptp()
                data['e%d' % g] = de

        self.summary_keys = ['e%d' % g for g in self.grid_points]


if __name__ == '__main__':
    from ase.tasks.main import run
    run(calcname='gpaw', task=EggboxTestTask())
