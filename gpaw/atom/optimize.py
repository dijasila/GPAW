from __future__ import print_function, division
import multiprocessing as mp
import glob
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
from ase.units import Bohr

from gpaw import GPAW, PW, setup_paths, Mixer  # , ConvergenceError
from gpaw.atom.generator2 import _generate, DatasetGenerationError


my_covalent_radii = covalent_radii.copy()
my_covalent_radii[1] += 0.2
my_covalent_radii[2] += 0.2
my_covalent_radii[10] += 0.2

NN = 10


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
                x = tuple(int(word) for word in words[:-10])
                self.individuals[x] = (error, n)
                y = tuple(float(word) for word in words[-10:])
                self.errors[n] = y
                
        self.fd = open('pool.csv', 'a')  # pool of genes
        self.n = len(self.individuals)
        self.pool = mp.Pool()  # process pool

    def run(self, func, sleep=5, mutate=15.0, size1=2, size2=1000):
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
            
            best = sorted(self.individuals.values())[:20]
            nbest = [N for e, N in best]
            for f in glob.glob('[0-9]*.txt'):
                if int(f[:-4]) not in nbest:
                    os.remove(f)
                    
            if len(self.individuals) > 40 and best[0][0] == np.inf:
                for result in results:
                    result.wait()
                return
                
    def new(self, mutate, size1, size2):
        if len(self.individuals) == 0:
            return self.initialvalue

        if len(self.individuals) < 32:
            x3 = np.array(self.initialvalue, dtype=float)
            mutate *= 2.5
        else:
            all = sorted((y, x)
                         for x, y in self.individuals.items()
                         if y is not None)  # and y != np.inf)
            S = len(all)
            if S < size1:
                x3 = np.array(self.initialvalue, dtype=float)
            else:
                parents = random.sample(all[:size1], 2)
                if S > size1 and random.random() < 0.33:
                    i = random.randint(size1, min(S, size2) - 1)
                    parents.append(all[i])
                    del parents[random.randint(0, 1)]
                else:
                    mutate /= 3
                    
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
    tolerances = np.array([0.01,
                           0.01, 0.03, 0.05,
                           0.01, 0.03, 0.05,
                           40, 0.2,
                           # 0.001, 0.02,
                           0.1])
    
    conf = None
    
    def __init__(self, symbol='H'):
    
        self.symbol = symbol

        with open('../start.txt') as fd:
            for line in fd:
                words = line.split()
                if words[1] == symbol:
                    projectors = words[3]
                    radii = [float(f) for f in words[5].split(',')]
                    r0 = float(words[7].split(',')[1])
                    break
            
        # Parse projectors string:
        pattern = r'(-?\d+\.\d)'
        energies = []
        for m in re.finditer(pattern, projectors):
            energies.append(float(projectors[m.start():m.end()]))
        self.projectors = re.sub(pattern, '%.1f', projectors)
        self.nenergies = len(energies)
        
        if 'f' in self.projectors:
            self.lmax = 3
        else:
            self.lmax = 2
            
        # Round to integers:
        x = ([e / 0.05 for e in energies] +
             [r / 0.01 for r in radii] +
             [r0 / 0.01])
        self.x = tuple(int(round(f)) for f in x)
        
        # Read FHI-Aims data:
        self.reference = {'fcc': read_reference('fcc', symbol),
                          'rocksalt': read_reference('rocksalt', symbol)}

        self.ecut1 = 450.0
        self.ecut2 = 800.0
        
        setup_paths[:0] = ['../..', '.']
        
        self.Z = atomic_numbers[symbol]
        self.rc = my_covalent_radii[self.Z]
        self.rco = my_covalent_radii[8]

    def run(self):  # , mu, n1, n2):
        # mu = float(mu)
        # n1 = int(n1)
        # n2 = int(n2)
        ga = GA(self.x)
        ga.run(self)  # , mutate=mu, size1=n1, size2=n2)
        
    def best(self, N=None):
        ga = GA(self.x)
        best = sorted((error, id, x)
                      for x, (error, id) in ga.individuals.items())
        if 0:
            import pickle
            pickle.dump(sorted(ga.individuals.values()), open('Zn.pckl', 'w'))
        if N is None:
            return len(best), best[0] + (ga.errors[best[0][1]],)
        else:
            return [(error, id, x, ga.errors[id])
                    for error, id, x in best[:N]]
        
    def summary(self, N=10):
        # print('dFffRrrICEer:')
        for error, id, x, errors in self.best(N):
            params = [0.05 * p for p in x[:self.nenergies]]
            params += [0.01 * p for p in x[self.nenergies:]]
            print('{0:5} {1:7.1f} {2} {3}'.format(
                id, error,
                ' '.join('{0:5.2f}'.format(p) for p in params),
                ' '.join('{0:8.3f}'.format(e) for e in errors)))
            
    def best1(self):
        try:
            n, (error, id, x, errors) = self.best()
        except IndexError:
            return
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
            self.generate(None, 'PBE', projectors, radii, r0, True, '',
                          logderivs=not False)
        
    def generate(self, fd, xc, projectors, radii, r0,
                 scalar_relativistic=False, tag=None, logderivs=True):
        if projectors[-1].isupper():
            nderiv0 = 5
        else:
            nderiv0 = 2
        gen = _generate(self.symbol, xc, self.conf, projectors, radii,
                        scalar_relativistic, None, r0, nderiv0,
                        ('poly', 4), None, None, fd)
        if not gen.check_all():
            raise DatasetGenerationError(xc)

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
        energies = tuple(0.05 * i for i in x[:self.nenergies])
        radii = [0.01 * i for i in x[self.nenergies:-1]]
        r0 = 0.01 * x[-1]
        projectors = self.projectors % energies
        return energies, radii, r0, projectors
        
    def __call__(self, n, x):
        fd = open('{0}.txt'.format(n), 'w')
        energies, radii, r0, projectors = self.parameters(x)
        
        if any(r < r0 for r in radii):  # or any(e <= 0.0 for e in energies):
            return n, x, [np.inf] * 10, np.inf
            
        try:
            errors = self.test(n, fd, projectors, radii, r0)
        except Exception:
            traceback.print_exc(file=fd)
            errors = [np.inf] * 10
            
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
        
        for name in ['slab', 'fcc', 'rocksalt']:  # , 'eggbox']:
            result = getattr(self, name)(n, fd)
            results[name] = result
            
        rc = self.rc / Bohr
        results['radii'] = sum(r - rc for r in radii if r > rc)
        
        errors = self.calculate_total_error(fd, results)
        
        return errors

    def calculate_total_error(self, fd, results):
        errors = [results['dataset']]
        maxiter = results['slab']
        
        for name in ['fcc', 'rocksalt']:
            result = results[name]
            maxiter = max(maxiter, result['maxiter'])
            errors.append(result['a'] - result['a0'])
            errors.append(result['c90'] - result['c90ref'])
            errors.append(result['c80'] - result['c80ref'])
        
        errors.append(maxiter)
        errors.append(results['fcc']['convergence'])
        
        # errors.append(results['eggbox'][0])
        # errors.append(results['eggbox'][1])
        
        errors.append(results['radii'])
        
        return errors
        
    def fcc(self, n, fd):
        ref = self.reference['fcc']
        a0r = ref['a']  # scalar-relativistic minimum
        sc = min(0.8, 2 * self.rc * 2**0.5 / a0r)
        sc = min((abs(s - sc), s) for s in ref if s != 'a')[1]
        maxiter = 0
        energies = []
        M = 200
        if 58 <= self.Z <= 70:
            M = 999
        for s in [sc, 0.9, 1.0, 1.1]:
            atoms = bulk(self.symbol, 'fcc', a0r * s)
            atoms.calc = GPAW(mode=PW(self.ecut2),
                              kpts={'density': 2.0, 'even': True},
                              xc='PBE',
                              lmax=self.lmax,
                              setups='ga' + str(n),
                              maxiter=M,
                              txt=fd)
            e = atoms.get_potential_energy()
            maxiter = max(maxiter, atoms.calc.get_number_of_iterations())
            energies.append(e)
            if s == 1.0:
                de = 0.0
                for ecut in [450, 550, 650]:
                    atoms.calc.set(mode=PW(self.ecut1), eigensolver='rmm-diis')
                    e2 = atoms.get_potential_energy()
                    de = max(de, abs(e2 - e))
                    maxiter = max(maxiter,
                                  atoms.calc.get_number_of_iterations())
                    
        return {'convergence': de,
                'c90': energies[1] - energies[2],
                'c80': energies[0] - energies[2],
                'c90ref': ref[0.9] - ref[1.0],
                'c80ref': ref[sc] - ref[1.0],
                'a0': fit([ref[s] for s in [0.9, 1.0, 1.1]]) * 0.1 * a0r,
                'a': fit(energies[1:]) * 0.1 * a0r,
                'maxiter': maxiter}
        
    def rocksalt(self, n, fd):
        ref = self.reference['rocksalt']
        a0r = ref['a']
        sc = min(0.8, 2 * (self.rc + self.rco) / a0r)
        sc = min((abs(s - sc), s) for s in ref if s != 'a')[1]
        maxiter = 0
        energies = []
        M = 200
        if 58 <= self.Z <= 70:
            M = 999
        for s in [sc, 0.9, 1.0, 1.1]:
            atoms = bulk(self.symbol + 'O', 'rocksalt', a0r * s)
            atoms.calc = GPAW(mode=PW(self.ecut2),
                              kpts={'density': 2.0, 'even': True},
                              xc='PBE',
                              lmax=self.lmax,
                              setups={self.symbol: 'ga' + str(n)},
                              maxiter=M,
                              txt=fd)
            e = atoms.get_potential_energy()
            maxiter = max(maxiter, atoms.calc.get_number_of_iterations())
            energies.append(e)
        
        return {'c90': energies[1] - energies[2],
                'c80': energies[0] - energies[2],
                'c90ref': ref[0.9] - ref[1.0],
                'c80ref': ref[sc] - ref[1.0],
                'a0': fit([ref[s] for s in [0.9, 1.0, 1.1]]) * 0.1 * a0r,
                'a': fit(energies[1:]) * 0.1 * a0r,
                'maxiter': maxiter}
        
    def slab(self, n, fd):
        a0 = self.reference['fcc']['a']
        atoms = fcc111(self.symbol, (1, 1, 7), a0, vacuum=3.5)
        assert not atoms.pbc[2]
        if 58 <= self.Z <= 70:
            mixer = {'mixer': Mixer(0.002, 3)}
        else:
            mixer = {}
        atoms.calc = GPAW(mode=PW(self.ecut1),
                          kpts={'density': 2.0, 'even': True},
                          xc='PBE',
                          lmax=self.lmax,
                          setups='ga' + str(n),
                          maxiter=900,
                          txt=fd,
                          **mixer)
        atoms.get_potential_energy()
        itrs = atoms.calc.get_number_of_iterations()
        return itrs
        
    def eggbox(self, n, fd):
        h = 0.19
        a0 = 16 * h
        atoms = Atoms(self.symbol, cell=(a0, a0, a0), pbc=True)
        atoms.calc = GPAW(h=h,
                          xc='PBE',
                          lmax=self.lmax,
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
            os.chdir(symbol)
            do = DatasetOptimizer(symbol)
        else:
            os.mkdir(symbol)
            os.chdir(symbol)
            do = DatasetOptimizer(symbol)
        do.run()  # *args[1:])
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
