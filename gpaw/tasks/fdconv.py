import optparse

import numpy as np
from ase.atoms import Atoms
from ase.tasks.task import Task
from ase.io import string2index
from ase.data import covalent_radii

from gpaw import FermiDirac


class ConvergenceTestTask(Task):
    taskname = 'fdconv'

    def __init__(self, g1=32, g2=40, L=6.4, **kwargs):
        """Calculate convergence of energy.

        The energy of a single atom and a dimer molecule is calculated
        for a range of grid-spacings."""

        self.grid_points = range(g1, g2 + 1, 4)
        
        self.L = L

        Task.__init__(self, calcfactory='gpaw', **kwargs)
        
    def build_system(self, name):
        return Atoms(name, pbc=True, cell=(self.L, self.L, self.L))

    def calculate(self, name, atoms):
        atoms.calc.set(occupations=FermiDirac(0.1),
                       kpts=[1, 1, 1])

        e1 = []
        for g in self.grid_points:
            atoms.calc.set(gpts=(g, g, g))
            e1.append(atoms.get_potential_energy())
        
        atoms += atoms
        atoms[1].position = [1.0, 1.1, 1.2]
        r = covalent_radii[atoms[0].number]
        atoms.set_distance(0, 1, 2 * r, 0)

        e2 = []
        for g in self.grid_points:
            atoms.calc.set(gpts=(g, g, g))
            e2.append(atoms.get_potential_energy())
        
        return {'e1': e1, 'e2': e2, 'grid points': self.grid_points}

    def analyse(self):
        self.summary_keys = []

        for name, data in self.data.items():
            if not data:
                continue
            grid_points = data['grid points']
            E1 = data['e1']
            E2 = data['e2']
            for g, e1, e2 in zip(grid_points, E1, E2)[:-1]:
                data['e%d' % g] = e1 - E1[-1]
                data['de%d' % g] = e1 - 0.5 * e2 - (E1[-1] -0.5 * E2[-1])

            if len(self.summary_keys) == 0:
                for g in grid_points[:-1]:
                    self.summary_keys.append('e%d' % g)
            
    def add_options(self, parser):
        Task.add_options(self, parser)

        grp = optparse.OptionGroup(parser, self.taskname.title())
        grp.add_option('-g', '--grid-point-range', default='32:41:4',
                       help='...')
        parser.add_option_group(grp)

    def parse(self, opts, args):
        Task.parse(self, opts, args)

        self.grid_points = range(100)[string2index(opts.grid_point_range)]


if __name__ == '__main__':
    from ase.tasks.main import run
    run(calcname='gpaw', task=ConvergenceTestTask())
