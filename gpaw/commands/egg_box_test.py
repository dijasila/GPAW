import numpy as np

from ase.cli.run import RunCommand

from gpaw import FermiDirac


class EggBoxTestCommand(RunCommand):
    def add_parser(self, subparser):
        parser = subparser.add_parser('egg-box-test', help='Egg-box ...')
        self.add_arguments(parser)
        
    def add_arguments(self, parser):
        RunCommand.add_arguments(self, parser)
        add = parser.add_argument
        add('-g', '--grid-spacings', default='0.18',
            help='...')

    def calculate(self, atoms, name):
        args = self.args
        
        atoms.pbc = True
        atoms.set_initial_magnetic_moments(None)
        atoms.calc.set(width=0.1,
                       kpts=[1, 1, 1])

        data = {'emax': [], 'fmax': [], 'energies': [], 'forces': [],
                'grid_spacings': []}
        for h in [float(x) for x in args.grid_spacings.split(',')]:
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
            if args.verbosity:
                print('%s %.3f Ang %.6f eV %.6f eV/Ang' % (name, h, emax, fmax))
            data['emax'].append(emax)
            data['fmax'].append(fmax)
            data['energies'].append(energies)
            data['forces'].append(forces)
            data['grid_spacings'].append(h)
        return data
