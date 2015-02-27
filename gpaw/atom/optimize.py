from __future__ import print_function, division
import multiprocessing as mp
import os
import random
import re
import time
import traceback

import numpy as np
from ase import Atoms
from ase.lattice import bulk
from ase.lattice.surface import fcc111

from gpaw import GPAW, PW, setup_paths
from gpaw.atom.generator2 import _generate


class GA:
    def __init__(self, filename, func, initialvalue=None):
        self.func = func
        self.initialvalue = initialvalue
        
        self.individuals = {}

        if os.path.isfile(filename):
            for line in open(filename):
                words = line.split(',')
                error = float(words.pop())
                n = int(words.pop(0))
                x = tuple(int(word) for word in words)
                self.individuals[x] = (error, n)
                
        self.fd = open(filename, 'a')
        self.n = len(self.individuals)
        self.pool = mp.Pool()

    def run(self, sleep=20, mutate=1.0, size1=2, size2=100):
        results = []
        while True:
            while len(results) < mp.cpu_count():
                x = self.new(mutate, size1, size2)
                self.individuals[x] = (None, self.n)
                result = self.pool.apply_async(self.func, [self.n, x])
                self.n += 1
                results.append(result)
            time.sleep(sleep)
            for result in results:
                if result.ready():
                    break
            else:
                continue
            results.remove(result)
            n, x, y = result.get()
            self.individuals[x] = (y, n)
            print('{0},{1},{2}'.format(n, ','.join(str(i) for i in x), y),
                  file=self.fd)
            self.fd.flush()
                
    def new(self, mutate, size1, size2):
        all = sorted((y, x) for x, y in self.individuals.items()
                     if y is not None)
        N = len(all)
        if N == 0:
            return self.initialvalue
        if N < size1:
            x3 = np.array(self.initialvalue)
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
    for line in open('energies_aims_{0}.csv'.format(name)):
        words = line.split(',')
        if words[0] == 'e':
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
    def __init__(self, symbol='H', projectors='1s,1.0s,0.0p,D',
                 radii=[0.9, 0.9], r0=0.8):
    
        self.symbol = symbol
        
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
        
        setup_paths[:0] = ['datasets']
        
    def run(self):
        ga = GA(self.symbol + '.csv', self, self.x)
        ga.run()
        
    def generate(self, fd, xc, projectors, radii, r0,
                 scalar_relativistic=False, tag=None):
        if projectors[-1].isupper():
            nderiv0 = 5
        else:
            nderiv0 = 2
        gen = _generate(self.symbol, xc, None, projectors, radii,
                        scalar_relativistic, None, r0, nderiv0,
                        ('poly', 4), None, None, fd)
        assert gen.check_all(), xc
        if tag:
            gen.make_paw_setup(tag).write_xml()
            name = '{0}.{1}.PBE'.format(self.symbol, tag)
            os.rename(name, 'datasets/' + name)
        r = 1.1 * gen.rcmax
        energies = np.linspace(-1.5, 2.0, 100)
        de = energies[1] - energies[0]
        error = 0.0
        for l in range(4):
            ld1 = gen.aea.logarithmic_derivative(l, energies, r)
            ld2 = gen.logarithmic_derivative(l, energies, r)
            error += ((ld1 - ld2)**2).sum() * de
        return error
            
    def __call__(self, n, x):
        fd = open('{0}.{1}.txt'.format(self.symbol, n), 'w')
        
        energies = tuple(0.1 * i for i in x[:self.nenergies])
        radii = [0.05 * i for i in x[self.nenergies:-1]]
        r0 = 0.05 * x[-1]
        
        fmt = 'PARAMS: E=[{0}] R=[{1}] r={2:.2f}'
        print(fmt.format(','.join('{0:.1f}'.format(e) for e in energies),
                         ','.join('{0:.2f}'.format(r) for r in radii),
                         r0), file=fd)
              
        projectors = self.projectors % energies
        
        try:
            error = self.test(n, fd, projectors, radii, r0)
        except Exception:
            traceback.print_exc(file=fd)
            error = np.inf
        return n, x, error
        
    def test(self, n, fd, projectors, radii, r0):
        error = self.generate(fd, 'PBE', projectors, radii, r0,
                              tag='ga{0}'.format(n))
        for xc in ['LDA', 'PBEsol']:
            error += self.generate(fd, xc, projectors, radii, r0,
                                   scalar_relativistic=True)
        results = {'dataset': error}
        for name in ['slab', 'fcc', 'rocksalt', 'eggbox']:
            result = getattr(self, name)(n, fd)
            results[name] = result
            
        errors = self.calculate_total_error(fd, results)
        
        return np.mean(errors)

    def calculate_total_error(self, fd, results):
        errors = [results['dataset'] / 3 / 5 / 0.1]
        
        maxiter = results['slab']
        
        for name in ['fcc', 'rocksalt']:
            result = results[name]
            maxiter = max(maxiter, result['maxiter'])
            errors.append(((result['a'] - result['a0']) / 0.02)**2)
            errors.append((result['de90'] / 0.01)**2)
            errors.append((result['de80'] / 0.03)**2)
        
        errors.append((maxiter / 30)**2)
        errors.append((results['fcc']['convergence'] / 0.1)**2)
        
        errors.append((results['eggbox'][0] / 0.001)**2)
        errors.append((results['eggbox'][1] / 0.02)**2)
        
        E = ''.join('{0:5.1f}{1}'.format(e, c)
                    for e, c in zip(errors, 'dFffRrrICEe'))
        print('ERRORS: {0:6.1f} {1} {2:6.1f}'.format(max(*errors),
                                                     E, sum(errors)),
              file=fd)
        return errors
        
    def fcc(self, n, fd, sizes=(0.8, 0.9, 1.0, 1.1)):
        ref = self.reference['fcc']
        a0r = ref['a']
        maxiter = 0
        energies = []
        for s in sizes:
            atoms = bulk(self.symbol, 'fcc', a0r * s)
            atoms.calc = GPAW(mode=PW(self.ecut2),
                              kpts={'density': 2.0, 'even': True},
                              xc='PBE',
                              setups='ga' + str(n),
                              maxiter=100,
                              txt=fd)
            e = atoms.get_potential_energy()
            maxiter = max(maxiter, atoms.calc.get_number_of_iterations())
            energies.append(e)
            if s == 1.0:
                atoms.calc.set(mode=PW(self.ecut1), eigensolver='rmm-diis')
                e2 = atoms.get_potential_energy()
                maxiter = max(maxiter, atoms.calc.get_number_of_iterations())

        return {'convergence': e2 - energies[2],
                'de80': energies[0] - energies[2] - (ref[0.8] - ref[1.0]),
                'de90': energies[1] - energies[2] - (ref[0.9] - ref[1.0]),
                'a0': fit([ref[s] for s in [0.9, 1.0, 1.1]]) * 0.1 * a0r,
                'a': fit(energies[1:]) * 0.1 * a0r,
                'maxiter': maxiter}
        
    def rocksalt(self, n, fd, sizes=(0.8, 0.9, 1.0, 1.1)):
        ref = self.reference['rocksalt']
        a0r = ref['a']
        maxiter = 0
        energies = []
        for s in sizes:
            atoms = bulk(self.symbol + 'O', 'rocksalt', a0r * s)
            atoms.calc = GPAW(mode=PW(self.ecut2),
                              kpts={'density': 2.0, 'even': True},
                              xc='PBE',
                              setups={self.symbol: 'ga' + str(n)},
                              maxiter=100,
                              txt=fd)
            e = atoms.get_potential_energy()
            maxiter = max(maxiter, atoms.calc.get_number_of_iterations())
            energies.append(e)
        
        return {'de80': energies[0] - energies[2] - (ref[0.8] - ref[1.0]),
                'de90': energies[1] - energies[2] - (ref[0.9] - ref[1.0]),
                'a0': fit([ref[s] for s in [0.9, 1.0, 1.1]]) * 0.1 * a0r,
                'a': fit(energies[1:]) * 0.1 * a0r,
                'maxiter': maxiter}
        
    def slab(self, n, fd):
        a0 = self.reference['fcc']['a']
        atoms = fcc111(self.symbol, (1, 1, 7), a0, vacuum=3.5)
        assert not atoms.pbc[2]
        atoms.calc = GPAW(mode=PW(self.ecut1),
                          kpts={'density': 2.0, 'even': True},
                          xc='PBE',
                          setups='ga' + str(n),
                          maxiter=100,
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
                          maxiter=100,
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
    # do = DatasetOptimizer()
    do = DatasetOptimizer('Cu', projectors='4s,1.0s,4p,1.0p,3d,1.0d',
                          radii=[2.1, 2.1, 2.0], r0=1.5)
    do.run()
