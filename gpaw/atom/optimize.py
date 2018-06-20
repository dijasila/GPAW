#!/bin/env python
from __future__ import print_function, division
import os
import re
import traceback

import numpy as np
from scipy.optimize import differential_evolution as DE
from ase import Atoms
from ase.data import covalent_radii, atomic_numbers, chemical_symbols
from ase.build import bulk
from ase.build import fcc111
from ase.units import Bohr

from gpaw import GPAW, PW, setup_paths, Mixer, ConvergenceError, Davidson
from gpaw.atom.generator2 import _generate  # , DatasetGenerationError


my_covalent_radii = covalent_radii.copy()
for e in ['Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']:  # missing radii
    my_covalent_radii[atomic_numbers[e]] = 1.7


NN = 5


class DatasetOptimizer:
    tolerances = np.array([0.3,
                           40,
                           2 / 3 * 300 * 0.1**0.25,  # convergence
                           0.0005,  # eggbox error
                           0.2])

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
                       [(0.2, 1.7 * self.rc / Bohr)] * (len(radii) + 1))

        self.ecut1 = 450.0
        self.ecut2 = 800.0

        setup_paths[:0] = ['.']

    def run(self):
        print('Running:', self.symbol, self.rc)
        # self.x = [3.96311677, 3.01924399, 0.98966047, 0.99035728, 0.84078698]
        self.x = [-1.7, 1.9, 1.6, 1.2, 1.2]
        self(self.x)
        #DE(self, self.bounds)

    def generate(self, fd, xc, projectors, radii, r0,
                 scalar_relativistic=False, tag=None, logderivs=True):
        if self.old:
            self.generate_old(fd, xc, scalar_relativistic, tag)
            return 0.0

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
            return np.inf

        if not scalar_relativistic:
            if not gen.check_all():
                print('dataset check failed')
                return np.inf

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
        print(x)
        fd = open('out.txt', 'w')
        energies, radii, r0, projectors = self.parameters(x)

        if any(r < r0 for r in radii):  # or any(e <= 0.0 for e in energies):
            # print(x, 'radii too small')
            return np.inf

        errors = self.test(fd, projectors, radii, r0)

        error = ((errors / self.tolerances)**2).sum()
        print(errors)
        print(error)
        return error

    def test(self, fd, projectors, radii, r0):
        error = self.generate(fd, 'PBE', projectors, radii, r0, tag='de')
        for xc in ['PBE', 'LDA', 'PBEsol', 'RPBE', 'PW91']:
            error += self.generate(fd, xc, projectors, radii, r0,
                                   scalar_relativistic=True)

        if not np.isfinite(error):
            return [np.inf] * NN

        results = {'dataset': error}

        try:
            results['convergence'], results['iters'] = self.convergence(fd)
            results['eggbox'] = self.eggbox(fd)
        except ConvergenceError:
            return np.inf
        except Exception as ex:
            print(ex)
            traceback.print_exc()
            return np.inf

        rc = self.rc / Bohr
        results['radii'] = sum(r - rc for r in radii if r > rc)

        errors = self.calculate_total_error(fd, results)

        return errors

    def calculate_total_error(self, fd, results):
        errors = [results['dataset']]
        iters = results['iters']

        for name in []:  # 'fcc', 'rocksalt']:
            result = results[name]
            if isinstance(result, dict):
                maxiter = max(maxiter, result['maxiter'])
                errors.append(result['a'] - result['a0'])
                errors.append(result['c90'] - result['c90ref'])
                errors.append(result['c80'] - result['c80ref'])
            else:
                maxiter = np.inf
                errors.extend([np.inf, np.inf, np.inf])

        errors.append(iters)
        errors.append(results['convergence'])

        errors.append(results['eggbox'])

        errors.append(results['radii'])

        return errors

    def slab(self, fd):
        a0 = self.reference['fcc']['a']
        atoms = fcc111(self.symbol, (1, 1, 7), a0, vacuum=3.5)
        assert not atoms.pbc[2]
        M = 333
        if 58 <= self.Z <= 70 or 90 <= self.Z <= 102:
            M = 1333
            mixer = {'mixer': Mixer(0.001, 3, 100)}
        else:
            mixer = {}
        atoms.calc = GPAW(mode=PW(self.ecut1),
                          kpts={'density': 2.0, 'even': True},
                          xc='PBE',
                          setups='de',
                          maxiter=M,
                          txt=fd,
                          **mixer)
        atoms.get_potential_energy()
        itrs = atoms.calc.get_number_of_iterations()
        return itrs

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
        print(energies)
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
            print(ec, de)
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

    def fcc(self, n, fd):
        ref = self.reference['fcc']
        a0r = ref['a']  # scalar-relativistic minimum
        sc = min(0.8, 2 * self.rc * 2**0.5 / a0r)
        sc = min((abs(s - sc), s) for s in ref if s != 'a')[1]
        maxiter = 0
        energies = []
        M = 200
        mixer = {}
        if 58 <= self.Z <= 70 or 90 <= self.Z <= 102:
            M = 999
            mixer = {'mixer': Mixer(0.01, 5)}
        for s in [sc, 0.9, 0.95, 1.0, 1.05]:
            atoms = bulk(self.symbol, 'fcc', a0r * s)
            atoms.calc = GPAW(mode=PW(self.ecut2),
                              kpts={'density': 4.0, 'even': True},
                              xc='PBE',
                              setups='ga' + str(n),
                              maxiter=M,
                              txt=fd,
                              **mixer)
            e = atoms.get_potential_energy()
            maxiter = max(maxiter, atoms.calc.get_number_of_iterations())
            energies.append(e)

        return {'c90': energies[1] - energies[3],
                'c80': energies[0] - energies[3],
                'c90ref': ref[0.9] - ref[1.0],
                'c80ref': ref[sc] - ref[1.0],
                'a0': fit([ref[s] for s in [0.95, 1.0, 1.05]]) * 0.05 * a0r,
                'a': fit(energies[2:]) * 0.05 * a0r,
                'maxiter': maxiter}

    def rocksalt(self, n, fd):
        ref = self.reference['rocksalt']
        a0r = ref['a']
        sc = min(0.8, 2 * (self.rc + self.rco) / a0r)
        sc = min((abs(s - sc), s) for s in ref if s != 'a')[1]
        maxiter = 0
        energies = []
        M = 200
        mixer = {}
        if 58 <= self.Z <= 70 or 90 <= self.Z <= 102:
            M = 999
            mixer = {'mixer': Mixer(0.01, 5)}
        for s in [sc, 0.9, 0.95, 1.0, 1.05]:
            atoms = bulk(self.symbol + 'O', 'rocksalt', a0r * s)
            atoms.calc = GPAW(mode=PW(self.ecut2),
                              kpts={'density': 4.0, 'even': True},
                              xc='PBE',
                              setups={self.symbol: 'ga' + str(n)},
                              maxiter=M,
                              txt=fd,
                              **mixer)
            e = atoms.get_potential_energy()
            maxiter = max(maxiter, atoms.calc.get_number_of_iterations())
            energies.append(e)

        return {'c90': energies[1] - energies[3],
                'c80': energies[0] - energies[3],
                'c90ref': ref[0.9] - ref[1.0],
                'c80ref': ref[sc] - ref[1.0],
                'a0': fit([ref[s] for s in [0.95, 1.0, 1.05]]) * 0.05 * a0r,
                'a': fit(energies[2:]) * 0.05 * a0r,
                'maxiter': maxiter}

    def best(self, N=None):
        ga = GA(self.x)
        best = sorted((error, id, x)
                      for x, (error, id) in ga.individuals.items())
        if 0:
            import pickle
            pickle.dump(sorted(ga.individuals.values()), open('Zn.pckl', 'wb'))
        if N is None:
            return len(best), best[0] + (ga.errors[best[0][1]],)
        else:
            return [(error, id, x, ga.errors[id])
                    for error, id, x in best[:N]]

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
        if not os.path.isdir(symbol):
            os.mkdir(symbol)
        os.chdir(symbol)
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
