import numpy as np
from ase.collections import g2
from gpaw import GPAW, LCAO
from ase.parallel import paropen
import time
from gpaw.directmin.etdm import ETDM

xc = 'PBE'
mode = LCAO()

with paropen('dm-g2-results.txt', 'w') as fd:
    for name in g2.names:
        atoms = g2[name]
        if len(atoms) == 1:
            continue
        atoms.center(vacuum=7.0)
        calc = GPAW(xc=xc, h=0.15,
                    convergence={'density': 1.0e-6},
                    basis='dzp',
                    mode=mode,
                    txt=name + '.txt',
                    eigensolver=ETDM(matrix_exp='egdecomp-u-invar',
                                     representation='u-invar'),
                    mixer={'backend': 'no-mixing'},
                    nbands='nao',
                    occupations={'name': 'fixed-uniform'},
                    symmetry='off'
                    )
        atoms.calc = calc

        t1 = time.time()
        e = atoms.get_potential_energy()
        t2 = time.time()
        steps = atoms.calc.get_number_of_iterations()
        iters = atoms.calc.wfs.eigensolver.eg_count
        print(name + "\t{}".format(iters),
              file=fd, flush=True)

output.splitlines()

# this is saved data
saved_data = {}
for i in output.splitlines():
    if i == '':
        continue
    mol = i.split()
    # ignore last two columns which are memory and elapsed time
    saved_data[mol[0]] = np.array([float(_) for _ in mol[1:-2]])

file2read = open('dm-g2-results.txt', 'r')
calculated_data_string = file2read.read().split('\n')
file2read.close()

# this is data calculated, we would like to coompare it to saved
# compare number of iteration, energy and gradient evaluation,
# and energy

calculated_data = {}
for i in calculated_data_string:
    if i == '':
        continue
    mol = i.split()
    # ignore last two columns which are memory and elapsed time
    calculated_data[mol[0]] = np.array([float(_) for _ in mol[1:-2]])

error = np.array([3, 3, 1.0e-3])

assert len(calculated_data) == len(saved_data)
for k in saved_data.keys():
    assert (abs(saved_data[k] - calculated_data[k]) < error).all()
