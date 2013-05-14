import numpy as np

from ase.cli.run import RunCommand

from gpaw.collections import fcc_data, rocksalt_data
from gpaw.wavefunctions.pw import PW


class BulkTestCommand(RunCommand):
    def add_parser(self, subparser):
        parser = subparser.add_parser('bulk-test', help='...')
        self.add_arguments(parser)

    def get_parameters(self):
        parameters = dict(kpts=3.0,
                          xc='PBE',
                          smearing=('Fermi-Dirac', 0.05),
                          eigensolver='cg')
        parameters.update(RunCommand.get_parameters(self))
        return parameters

    def calculate(self, atoms, name):
        args = self.args
        atoms.pbc = True
        b = fcc_data[name][0] / 2
        atoms.cell = [(0, b, b), (b, 0, b), (b, b, 0)]
        atoms.positions[0] = (0, 0, 0)
        atoms.set_initial_magnetic_moments(None)
        energies = []
        for ecut in [400, 600, 800]:
            atoms.calc.set(mode=PW(ecut))
            e = atoms.get_potential_energy()
            energies.append(e)
        atoms.cell *= 0.85
        e = atoms.get_potential_energy()
        energies.append(e)
        b = rocksalt_data[name][0] / 2
        atoms.cell = [(0, b, b), (b, 0, b), (b, b, 0)]
        atoms.append('O')
        atoms.positions[1] = (b, 0, 0)
        e = atoms.get_potential_energy()
        energies.append(e)
        atoms.cell *= 0.85
        e = atoms.get_potential_energy()
        energies.append(e)
        return {'energies': energies}
