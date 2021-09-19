import numpy as np
from ase.collections import g2
from doc.documentation.directmin import tests_data
from gpaw import GPAW, LCAO
from ase.parallel import paropen
import time
from gpaw.directmin.etdm import ETDM


def read_saved_data(output):
    saved_data = {}
    for i in output.splitlines():
        if i == '':
            continue
        mol = i.split()
        saved_data[mol[0]] = np.array([float(_) for _ in mol[1:]])

    return saved_data


# Results (total energy, number of iterations) obtained
# in a previous calculation. Used to compare with the
# current results.
saved_data_dm = read_saved_data(tests_data.output_g2_with_dm)

xc = 'PBE'
mode = LCAO()

with paropen('dm-g2-results.txt', 'w') as fd:
    for name in saved_data_dm.keys():
        atoms = g2[name]
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
        assert abs(saved_data_dm[name][0] - e) < 1.0e-2

        steps = atoms.calc.get_number_of_iterations()
        iters = atoms.calc.wfs.eigensolver.eg_count
        assert abs(saved_data_dm[name][1] - iters) < 3

        print(name + "\t{}".format(iters),
              file=fd, flush=True)
