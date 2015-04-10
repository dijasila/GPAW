from __future__ import print_function, division
import multiprocessing as mp
import os
import random
import re
import time
import traceback

import numpy as np
from ase import Atoms
from ase.data import covalent_radii, atomic_numbers
from ase.lattice import bulk
from ase.lattice.surface import fcc111

from gpaw import GPAW, PW, setup_paths, ConvergenceError
from gpaw.atom.generator2 import _generate, DatasetGenerationError


class GA:
    def __init__(self, initialvalue=None):
        self.initialvalue = initialvalue
        
        self.individuals = {}
        self.errors = {}
        
        if os.path.isfile('pool.csv'):
            for line in open('pool.csv'):
                words = line.split(',')
                n = int(words.pop(0))
                error = float(words.pop(0))
                x = tuple(int(word) for word in words[:-9])
                self.individuals[x] = (error, n)
                y = tuple(float(word) for word in words[-9:])
                self.errors[n] = y
                
        self.fd = open('pool.csv', 'a')  # pool of genes
        self.n = len(self.individuals)
        self.pool = mp.Pool()  # process pool

    def run(self, func, sleep=20, mutate=3.0, size1=2, size2=100):
        results = []
        while True:
            while len(results) < mp.cpu_count():
                x = self.new(mutate, size1, size2)
                self.individuals[x] = (None, self.n)
                result = self.pool.apply_async(func, [self.n, x])
                self.n += 1
                results.append(result)
            time.sleep(sleep)
            for result in results:
                if result.ready():
                    break
            else:
                continue
            results.remove(result)
            n, x, errors, error = result.get()
            self.individuals[x] = (error, n)
            print('{0},{1},{2},{3}'.format(n, error,
                                           ','.join(str(i) for i in x),
                                           ','.join('{0:.4f}'.format(e)
                                                    for e in errors)),
                  file=self.fd)
            self.fd.flush()
                
    def new(self, mutate, size1, size2):
        all = sorted((y, x) for x, y in self.individuals.items()
                     if y is not None)
        N = len(all)
        if N == 0:
            return self.initialvalue
        if N < size1:
            x3 = np.array(self.initialvalue, dtype=float)
        else:
            parents = random.sample(all[:size1], 2)
            if N > size1:
                i = random.randint(size1, min(N, size2) - 1)
                parents.append(all[i])
                del parents[random.randint(0, 2)]
                
            x1 = parents[0][1]
            x2 = parents[1][1]
            r = np.random.rand(len(x1))
            x3 = r * x1 + (1 - r) * x2

        while True:
            x3 += np.random.normal(0, mutate, len(x3))
            x = tuple(int(round(a)) for a in x3)
            if x not in self.individuals:
                break
        
        return x


def read_reference(name, symbol):
    for line in open('../../{0}.csv'.format(name)):
        words = line.split(',')
        if words[0] == 'c':
            x = [float(word) for word in words[2:]]
        elif words[0] == symbol:
            ref = dict((c, float(word))
                       for c, word in zip(x, words[2:]))
            ref['a'] = float(words[1])
            return ref
                
        
def fit(E):
    em, e0, ep = E
    a = (ep + em) / 2 - e0
    b = (ep - em) / 2
    return -b / (2 * a)
    
    
class DatasetOptimizer:
    tolerances = np.array([0.1, 0.01, 0.1, 0.01, 0.1, 40, 0.2, 0.001, 0.02])
    
    def __init__(self, symbol='H', projectors=None,
                 radii=None, r0=None):
    
        self.symbol = symbol

        if os.path.isfile('parameters.txt'):
            with open('parameters.txt') as fd:
                words = fd.readline().split()
                projectors = words.pop(0)
                radii = [float(f) for f in words]
                r0 = radii.pop()
        else:
            with open('parameters.txt', 'w') as fd:
                print(projectors, ' '.join('{0:.2f}'.format(r)
                                           for r in radii + [r0]),
                      file=fd)
            
        # Parse projectors string:
        pattern = r'(-?\d+\.\d)'
        energies = []
        for m in re.finditer(pattern, projectors):
            energies.append(float(projectors[m.start():m.end()]))
        self.projectors = re.sub(pattern, '%.1f', projectors)
        self.nenergies = len(energies)
        
        # Round to integers:
        x = ([e / 0.1 for e in energies] +
             [r / 0.05 for r in radii] +
             [r0 / 0.05])
        self.x = tuple(int(round(f)) for f in x)
        
        # Read FHI-Aims data:
        self.reference = {'fcc': read_reference('fcc', symbol),
                          'rocksalt': read_reference('rocksalt', symbol)}

        self.ecut1 = 400.0
        self.ecut2 = 800.0
        
        setup_paths[:0] = ['../..', '.']
        
        Z = atomic_numbers[symbol]
        self.rc = covalent_radii[Z]
        self.rco = covalent_radii[8]

    def run(self):
        ga = GA(self.x)
        ga.run(self)
        
    def best(self, N=None):
        ga = GA(self.x)
        best = sorted((error, id, x)
                      for x, (error, id) in ga.individuals.items())
        if N is None:
            return best[0] + (ga.errors[best[0][1]],)
        else:
            return [(error, id, x, ga.errors[id])
                    for error, id, x in best[:N]]
        
    def summary(self, N=10):
        print('dFffRrrICEe:')
        for error, id, x, errors in self.best(N):
            params = [0.1 * p for p in x[:self.nenergies]]
            params += [0.05 * p for p in x[self.nenergies:]]
            print('{0:5} {1:7.1f} {2} {3}'.format(
                id, error,
                ' '.join('{0:5.2f}'.format(p) for p in params),
                ' '.join('{0:8.3f}'.format(e) for e in errors)))
            
    def best1(self):
        error, id, x, errors = self.best()
        energies, radii, r0, projectors = self.parameters(x)
        print('ERROR:', self.symbol, error)
        print('ERRORS:', self.symbol, errors)
        print('PARAMS:', self.symbol, energies, radii, r0, projectors)
        print(self.symbol, self.rc / 0.53,
              ''.join('{0:7.2f}'.format(r * 0.53 / self.rc)
                      for r in radii + [r0]))
        if 0:
            self.generate(None, 'PBE', projectors, radii, r0, not True, '',
                          logderivs=False)
        
    def generate(self, fd, xc, projectors, radii, r0,
                 scalar_relativistic=False, tag=None, logderivs=True):
        if projectors[-1].isupper():
            nderiv0 = 5
        else:
            nderiv0 = 2
        gen = _generate(self.symbol, xc, None, projectors, radii,
                        scalar_relativistic, None, r0, nderiv0,
                        ('poly', 4), None, None, fd)
        assert gen.check_all(), xc

        if tag is not None:
            gen.make_paw_setup(tag or None).write_xml()

        r = 1.1 * gen.rcmax
        energies = np.linspace(-1.5, 2.0, 100)
        de = energies[1] - energies[0]
        error = 0.0
        if logderivs:
            for l in range(4):
                ld1 = gen.aea.logarithmic_derivative(l, energies, r)
                ld2 = gen.logarithmic_derivative(l, energies, r)
                error = max(error, abs(ld1 - ld2).sum() * de)
        return error
            
    def parameters(self, x):
        energies = tuple(0.1 * i for i in x[:self.nenergies])
        radii = [0.05 * i for i in x[self.nenergies:-1]]
        r0 = 0.05 * x[-1]
        projectors = self.projectors % energies
        return energies, radii, r0, projectors
        
    def __call__(self, n, x):
        fd = open('{0}.txt'.format(os.getpid()), 'w')
        
        energies, radii, r0, projectors = self.parameters(x)
        
        if not all(r0 <= r <= self.rc for r in radii):
            return n, x, [np.inf] * 9, np.inf
            
        try:
            errors = self.test(n, fd, projectors, radii, r0)
        except (ConvergenceError, DatasetGenerationError):
            traceback.print_exc(file=fd)
            errors = [np.inf] * 9
            
        try:
            os.remove('{0}.ga{1}.PBE'.format(self.symbol, n))
        except OSError:
            pass
        
        return n, x, errors, ((errors / self.tolerances)**2).sum()
        
    def test(self, n, fd, projectors, radii, r0):
        error = self.generate(fd, 'PBE', projectors, radii, r0,
                              tag='ga{0}'.format(n))
        for xc in ['PBE', 'LDA', 'PBEsol']:
            error += self.generate(fd, xc, projectors, radii, r0,
                                   scalar_relativistic=True)
        results = {'dataset': error}
        for name in ['slab', 'fcc', 'rocksalt', 'eggbox']:
            result = getattr(self, name)(n, fd)
            results[name] = result
            
        errors = self.calculate_total_error(fd, results)
        
        return errors

    def calculate_total_error(self, fd, results):
        errors = [results['dataset']]
        maxiter = results['slab']
        
        for name in ['fcc', 'rocksalt']:
            result = results[name]
            maxiter = max(maxiter, result['maxiter'])
            errors.append(result['a'] - result['a0'])
            errors.append(result['de'])
        
        errors.append(maxiter)
        errors.append(results['fcc']['convergence'])
        
        errors.append(results['eggbox'][0])
        errors.append(results['eggbox'][1])
        
        return errors
        
    def fcc(self, n, fd):
        ref = self.reference['fcc']
        a0r = ref['a']  # scalar-relativistic minimum
        sc = min(0.8, 2 * self.rc * 2**0.5 / a0r)
        sc = min((abs(s - sc), s) for s in ref if s != 'a')[1]
        maxiter = 0
        energies = []
        for s in [sc, 0.95, 1.0, 1.05]:
            atoms = bulk(self.symbol, 'fcc', a0r * s)
            atoms.calc = GPAW(mode=PW(self.ecut2),
                              kpts={'density': 2.0, 'even': True},
                              xc='PBE',
                              setups='ga' + str(n),
                              maxiter=200,
                              txt=fd)
            e = atoms.get_potential_energy()
            maxiter = max(maxiter, atoms.calc.get_number_of_iterations())
            energies.append(e)
            if s == 1.0:
                atoms.calc.set(mode=PW(self.ecut1), eigensolver='rmm-diis')
                e2 = atoms.get_potential_energy()
                maxiter = max(maxiter, atoms.calc.get_number_of_iterations())

        return {'convergence': e2 - energies[2],
                'de': energies[0] - energies[2] - (ref[sc] - ref[1.0]),
                'a0': fit([ref[s] for s in [0.95, 1.0, 1.05]]) * 0.05 * a0r,
                'a': fit(energies[1:]) * 0.05 * a0r,
                'maxiter': maxiter}
        
    def rocksalt(self, n, fd):
        ref = self.reference['rocksalt']
        a0r = ref['a']
        sc = min(0.8, (self.rc + self.rco) / a0r)
        sc = min((abs(s - sc), s) for s in ref if s != 'a')[1]
        maxiter = 0
        energies = []
        for s in [sc, 0.95, 1.0, 1.05]:
            atoms = bulk(self.symbol + 'O', 'rocksalt', a0r * s)
            atoms.calc = GPAW(mode=PW(self.ecut2),
                              kpts={'density': 2.0, 'even': True},
                              xc='PBE',
                              setups={self.symbol: 'ga' + str(n)},
                              maxiter=200,
                              txt=fd)
            e = atoms.get_potential_energy()
            maxiter = max(maxiter, atoms.calc.get_number_of_iterations())
            energies.append(e)
        
        return {'de': energies[0] - energies[2] - (ref[sc] - ref[1.0]),
                'a0': fit([ref[s] for s in [0.95, 1.0, 1.05]]) * 0.05 * a0r,
                'a': fit(energies[1:]) * 0.05 * a0r,
                'maxiter': maxiter}
        
    def slab(self, n, fd):
        a0 = self.reference['fcc']['a']
        atoms = fcc111(self.symbol, (1, 1, 7), a0, vacuum=3.5)
        assert not atoms.pbc[2]
        atoms.calc = GPAW(mode=PW(self.ecut1),
                          kpts={'density': 2.0, 'even': True},
                          xc='PBE',
                          setups='ga' + str(n),
                          maxiter=200,
                          txt=fd)
        atoms.get_potential_energy()
        itrs = atoms.calc.get_number_of_iterations()
        return itrs
        
    def eggbox(self, n, fd):
        h = 0.19
        a0 = 16 * h
        atoms = Atoms(self.symbol, cell=(a0, a0, a0), pbc=True)
        atoms.calc = GPAW(h=h,
                          xc='PBE',
                          symmetry='off',
                          setups='ga' + str(n),
                          maxiter=200,
                          txt=fd)
        e0 = atoms.get_potential_energy()
        atoms.positions += h / 6
        e1 = atoms.get_potential_energy()
        f1 = atoms.get_forces()
        atoms.positions += h / 6
        e2 = atoms.get_potential_energy()
        f2 = atoms.get_forces()
        atoms.positions += h / 6
        e3 = atoms.get_potential_energy()
        return (np.ptp([e0, e1, e2, e3]),
                max((f**2).sum()**0.5 for f in [f1, f2]))

        
if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser(usage='python -m gpaw.atom.optimize '
                                   '[options] element',
                                   description='Optimize dataset')
    parser.add_option('-s', '--summary', action='store_true')
    parser.add_option('-b', '--best', action='store_true')
    parser.add_option('-r', '--run', action='store_true')
    opts, args = parser.parse_args()
    if opts.run:
        symbol = args[0]
        if os.path.isdir(symbol):
            do = DatasetOptimizer(symbol)
            os.chdir(symbol)
        else:
            os.mkdir(symbol)
            os.chdir(symbol)
            projectors, radii, r0 = args[1:]
            radii = [float(r) for r in radii.split(',')]
            r0 = float(r0)
            do = DatasetOptimizer(symbol, projectors, radii, r0)
        do.run()
    else:
        if len(args) == 0:
            symbol = os.getcwd().rsplit('/', 1)[1]
            args.append(symbol)
            os.chdir('..')
        for symbol in args:
            os.chdir(symbol)
            do = DatasetOptimizer(symbol)
            if opts.summary:
                do.summary()
            elif opts.best:
                do.best1()
            os.chdir('..')
