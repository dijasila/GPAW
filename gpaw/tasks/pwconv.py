import optparse

import numpy as np
from ase.atoms import Atoms
from ase.tasks.task import Task
from ase.io import string2index
from ase.data import covalent_radii

from gpaw import FermiDirac, PW


class ConvergenceTestTask(Task):
    taskname = 'pwconv'

    def __init__(self, L=6.0, cutoffs=(300, 400, 500), **kwargs):
        """Calculate convergence of energy.

        The energy of a single atom and a dimer molecule is calculated
        for a range of plane-wave cutoffs."""

        self.cutoffs = cutoffs
        self.L = L

        Task.__init__(self, calcfactory='gpaw', **kwargs)

    def build_system(self, name):
        return Atoms(name, pbc=True, cell=(self.L, self.L, self.L))

    def calculate(self, name, atoms):
        atoms.calc.set(occupations=FermiDirac(0.1),
                       kpts=[1, 1, 1],
                       gpts=None)

        e1 = []
        for e in self.cutoffs:
            atoms.calc.density = None
            atoms.calc.hamiltonian = None
            atoms.calc.set(mode=PW(e))
            e1.append(atoms.get_potential_energy())
        
        atoms += atoms
        atoms[1].position = [1.0, 1.1, 1.2]
        r = covalent_radii[atoms[0].number]
        atoms.set_distance(0, 1, 2 * r, 0)

        e2 = []
        for e in self.cutoffs:
            atoms.calc.density = None
            atoms.calc.hamiltonian = None
            atoms.calc.set(mode=PW(e))
            e2.append(atoms.get_potential_energy())
        
        return {'e1': e1, 'e2': e2, 'cutoffs': self.cutoffs}

    def analyse(self):
        self.summary_keys = []

        for name, data in self.data.items():
            cutoffs = data['cutoffs']
            E1 = data['e1']
            E2 = data['e2']
            for ecut, e1, e2 in zip(cutoffs, E1, E2)[:-1]:
                data['e%d' % ecut] = e1 - E1[-1]
                data['de%d' % ecut] = e1 - 0.5 * e2 - (E1[-1] -0.5 * E2[-1])

            if len(self.summary_keys) == 0:
                for ecut in cutoffs[:-1]:
                    self.summary_keys.append('e%d' % ecut)
        
    def add_options(self, parser):
        Task.add_options(self, parser)

        grp = optparse.OptionGroup(parser, self.taskname.title())
        grp.add_option('--ecut', default='300:700:100',
                       help='...')
        parser.add_option_group(grp)

    def parse(self, opts, args):
        Task.parse(self, opts, args)

        e1, e2, de = (float(x) for x in opts.ecut.split(':'))
        self.cutoffs = np.linspace(e1, e2, round((e2 - e1) / de) + 1)


if __name__ == '__main__':
    from ase.tasks.main import run
    run(calcname='gpaw', task=ConvergenceTestTask())
