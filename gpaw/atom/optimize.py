from __future__ import print_function, division
import os
import re

import numpy as np
from scipy.optimize import differential_evolution as DE
from ase import Atoms
from ase.data import covalent_radii, atomic_numbers, chemical_symbols
from ase.units import Bohr

from gpaw import GPAW, PW, setup_paths, Mixer, ConvergenceError, Davidson
from gpaw.atom.generator2 import _generate  # , DatasetGenerationError


my_covalent_radii = covalent_radii.copy()
for e in ['Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']:  # missing radii
    my_covalent_radii[atomic_numbers[e]] = 1.7


class PAWDataError(Exception):
    """Error in PAW data generation."""


class DatasetOptimizer:
    tolerances = np.array([0.2,
                           0.3,
                           40,
                           2 / 3 * 300 * 0.1**0.25,  # convergence
                           0.0005])  # eggbox error

    def __init__(self, symbol='H', nc=False, processes=None):
        self.old = False

        self.symbol = symbol
        self.nc = nc
        self.processes = processes

        with open('../start.txt') as fd:
            for line in fd:
                words = line.split()
                if words[1] == symbol:
                    projectors = words[3]
                    radii = [float(f) for f in words[5].split(',')]
                    r0 = float(words[7].split(',')[1])
                    break
            else:
                raise ValueError

        self.Z = atomic_numbers[symbol]
        self.rc = my_covalent_radii[self.Z]

        # Parse projectors string:
        pattern = r'(-?\d+\.\d)'
        energies = []
        for m in re.finditer(pattern, projectors):
            energies.append(float(projectors[m.start():m.end()]))
        self.projectors = re.sub(pattern, '{:.1f}', projectors)
        self.nenergies = len(energies)

        # Round to integers:
        self.x = energies + radii + [r0]
        self.bounds = ([(-1.0, 4.0)] * self.nenergies +
                       [(r * 0.6, r * 1.5) for r in radii] +
                       [(0.3, max(radii) * 1.4)])

        self.ecut1 = 450.0
        self.ecut2 = 800.0

        setup_paths[:0] = ['.']

    def run(self):
        print(self.symbol, self.rc / Bohr)
        print(self.x)
        print(self.bounds)
        DE(self, self.bounds)

    def generate(self, fd, projectors, radii, r0, xc,
                 scalar_relativistic=True, tag=None, logderivs=True):

        if projectors[-1].isupper():
            nderiv0 = 5
        else:
            nderiv0 = 2

        type = 'poly'
        if self.nc:
            type = 'nc'

        try:
            gen = _generate(self.symbol, xc, None, projectors, radii,
                            scalar_relativistic, None, r0, nderiv0,
                            (type, 4), None, None, fd)
        except np.linalg.linalg.LinAlgError:
            raise PAWDataError('LinAlgError')

        if not scalar_relativistic:
            if not gen.check_all():
                raise PAWDataError('dataset check failed')

        if tag is not None:
            gen.make_paw_setup(tag or None).write_xml()

        r = 1.1 * gen.rcmax

        lmax = 2
        if 'f' in projectors:
            lmax = 3

        error = 0.0
        if logderivs:
            for l in range(lmax + 1):
                emin = -1.5
                emax = 2.0
                n0 = gen.number_of_core_states(l)
                if n0 > 0:
                    e0_n = gen.aea.channels[l].e_n
                    emin = max(emin, e0_n[n0 - 1] + 0.1)
                energies = np.linspace(emin, emax, 100)
                de = energies[1] - energies[0]
                ld1 = gen.aea.logarithmic_derivative(l, energies, r)
                ld2 = gen.logarithmic_derivative(l, energies, r)
                error += abs(ld1 - ld2).sum() * de

        return error

    def parameters(self, x):
        energies = x[:self.nenergies]
        radii = x[self.nenergies:-1]
        r0 = x[-1]
        projectors = self.projectors.format(*energies)
        return energies, radii, r0, projectors

    def __call__(self, x):
        energies, radii, r0, projectors = self.parameters(x)

        print('({}, {}, {:.2f}):'
              .format(', '.join('{:+.2f}'.format(e) for e in energies),
                      ', '.join('{:.2f}'.format(r) for r in radii),
                      r0),
              end='')

        fd = open('out.txt', 'w')
        errors, msg = self.test(fd, projectors, radii, r0)
        error = ((errors / self.tolerances)**2).sum()

        print('{:9.1f} ({:.2f}, {:.3f}, {:.1f}, {}, {:.5f}) {}'
              .format(error, *errors, msg),
              flush=True)

        return error

    def test(self, fd, projectors, radii, r0):
        errors = [np.inf] * 5

        try:
            if any(r < r0 for r in radii):
                raise PAWDataError('Core radius too large')

            rc = self.rc / Bohr
            errors[0] = sum(r - rc for r in radii if r > rc)

            error = 0.0
            for kwargs in [dict(xc='PBE', tag='de'),
                           dict(xc='PBE', scalar_relativistic=False),
                           dict(xc='LDA'),
                           dict(xc='PBEsol'),
                           dict(xc='RPBE'),
                           dict(xc='PW91')]:
                error += self.generate(fd, projectors, radii, r0, **kwargs)
            errors[1] = error

            errors[2:4] = self.convergence(fd)

            errors[4] = self.eggbox(fd)

        except (ConvergenceError, PAWDataError) as e:
            msg = str(e)
        else:
            msg = ''

        return errors, msg

    def eggbox(self, fd):
        energies = []
        for h in [0.16, 0.18, 0.2]:
            a0 = 16 * h
            atoms = Atoms(self.symbol, cell=(a0, a0, 2 * a0), pbc=True)
            M = 333
            if 58 <= self.Z <= 70 or 90 <= self.Z <= 102:
                M = 999
                mixer = {'mixer': Mixer(0.01, 5)}
            else:
                mixer = {}
            atoms.calc = GPAW(h=h,
                              eigensolver=Davidson(niter=2),
                              xc='PBE',
                              symmetry='off',
                              setups='de',
                              maxiter=M,
                              txt=fd,
                              *mixer)
            atoms.positions += h / 2  # start with broken symmetry
            e0 = atoms.get_potential_energy()
            atoms.positions -= h / 6
            e1 = atoms.get_potential_energy()
            atoms.positions -= h / 6
            e2 = atoms.get_potential_energy()
            atoms.positions -= h / 6
            e3 = atoms.get_potential_energy()
            energies.append(np.ptp([e0, e1, e2, e3]))
        # print(energies)
        return max(energies)

    def convergence(self, fd):
        a = 3.0
        atoms = Atoms(self.symbol, cell=(a, a, a), pbc=True)
        M = 333
        if 58 <= self.Z <= 70 or 90 <= self.Z <= 102:
            M = 999
            mixer = {'mixer': Mixer(0.01, 5)}
        else:
            mixer = {}
        atoms.calc = GPAW(mode=PW(1500),
                          xc='PBE',
                          setups='de',
                          symmetry='off',
                          maxiter=M,
                          txt=fd,
                          **mixer)
        e0 = atoms.get_potential_energy()
        iters = atoms.calc.get_number_of_iterations()
        oldfde = None
        area = 0.0

        def f(x):
            return x**0.25

        for ec in range(800, 200, -100):
            atoms.calc.set(mode=PW(ec))
            atoms.calc.set(eigensolver='rmm-diis')
            de = atoms.get_potential_energy() - e0
            # print(ec, de)
            fde = f(abs(de))
            if fde > f(0.1):
                if oldfde is None:
                    return np.inf, iters
                ec0 = ec + (fde - f(0.1)) / (fde - oldfde) * 100
                area += ((ec + 100) - ec0) * (f(0.1) + oldfde) / 2
                break

            if oldfde is not None:
                area += 100 * (fde + oldfde) / 2
            oldfde = fde

        return area, iters

    def summary(self, N=10):
        # print('dFffRrrICEer:')
        for error, id, x, errors in self.best(N):
            params = [0.1 * p for p in x[:self.nenergies]]
            params += [0.05 * p for p in x[self.nenergies:]]
            print('{0:2} {1:2}{2:4}{3:6.1f}|{4}|'
                  '{5:4.1f}|'
                  # '{6:6.2f} {7:6.2f} {8:6.2f}|'
                  # '{9:6.2f} {10:6.2f} {11:6.2f}|'
                  # '{12:3.0f} {13:4.0f} {14:7.4f} {15:4.1f}'
                  '{6:3.0f} {7:4.0f} {8:7.4f} {9:4.1f}'
                  .format(self.Z,
                          self.symbol,
                          id, error,
                          ' '.join('{0:5.2f}'.format(p) for p in params),
                          *errors))

    def best1(self):
        try:
            n, (error, id, x, errors) = self.best()
        except IndexError:
            return  # self.Z, (np.nan, np.nan, np.nan, np.nan, np.nan)
        # return self.Z, errors / self.tolerances

        energies, radii, r0, projectors = self.parameters(x)
        if 0:
            print(error, self.symbol, n)
        if 1:
            if projectors[-1].isupper():
                nderiv0 = 5
            else:
                nderiv0 = 2
            fmt = '{0:2} {1:2} -P {2:31} -r {3:20} -0 {4},{5:.2f} # {6:10.3f}'
            print(fmt.format(self.Z,
                             self.symbol,
                             projectors,
                             ','.join('{0:.2f}'.format(r) for r in radii),
                             nderiv0,
                             r0,
                             error))
        if 0:
            with open('parameters.txt', 'w') as fd:
                print(projectors, ' '.join('{0:.2f}'.format(r)
                                           for r in radii + [r0]),
                      file=fd)
        if 0 and error != np.inf and error != np.nan:
            self.generate(None, 'PBE', projectors, radii, r0, True, 'v2e',
                          logderivs=False)


if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser(usage='python -m gpaw.atom.optimize '
                                   '[options] element',
                                   description='Optimize dataset')
    parser.add_option('-s', '--summary', action='store_true')
    parser.add_option('-b', '--best', action='store_true')
    parser.add_option('-r', '--run', action='store_true')
    parser.add_option('-m', '--minimize', action='store_true')
    parser.add_option('-n', '--norm-conserving', action='store_true')
    parser.add_option('-i', '--initial-only', action='store_true')
    parser.add_option('-o', '--old-setups', action='store_true')
    parser.add_option('-N', '--processes', type=int)
    opts, args = parser.parse_args()
    if opts.run or opts.minimize:
        symbol = args[0]
        do = DatasetOptimizer(symbol, opts.norm_conserving, opts.processes)
        if opts.run:
            if opts.initial_only:
                do.old = opts.old_setups
                do.run_initial()
            else:
                do.run()
        else:
            do.minimize()
    else:
        if args == ['.']:
            symbol = os.getcwd().rsplit('/', 1)[1]
            args = [symbol]
            os.chdir('..')
        elif len(args) == 0:
            args = [symbol for symbol in chemical_symbols
                    if os.path.isdir(symbol)]
        x = []
        y = []
        for symbol in args:
            os.chdir(symbol)
            try:
                do = DatasetOptimizer(symbol, opts.norm_conserving,
                                      opts.processes)
            except ValueError:
                pass
            else:
                if opts.summary:
                    do.summary(15)
                elif opts.best:
                    # a,b=
                    do.best1()
                    # x.append(a)
                    # y.append(b)
            os.chdir('..')
        if 0:
            import matplotlib.pyplot as plt
            for z, t in zip(np.array(y).T, 'LIPER'):
                plt.plot(x, z, label=t)
            plt.legend()
            plt.show()
