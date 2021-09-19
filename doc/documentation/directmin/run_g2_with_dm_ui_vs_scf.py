import numpy as np
from ase.collections import g2
from doc.documentation.directmin import tools_and_data
from gpaw import GPAW, LCAO, FermiDirac, ConvergenceError
from ase.parallel import paropen
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
saved_data = {0: read_saved_data(tools_and_data.output_g2_with_dm),
              1: read_saved_data(tools_and_data.output_g2_with_dm)}

calc_args = {'xc': 'PBE', 'h': 0.15,
             'convergence': {'density': 1.0e-6},
             'maxiter': 333, 'basis': 'dzp',
             'mode': LCAO(), 'symmetry': 'off'}

with paropen('dm-g2-results.txt', 'w') as fd:
    for name in saved_data[0].keys():
        atoms = g2[name]
        atoms.center(vacuum=7.0)
        for dm in [0, 1]:
            if dm:
                calc = GPAW(**calc_args,
                            txt=name + '_dm.txt',
                            eigensolver=ETDM(matrix_exp='egdecomp-u-invar',
                                             representation='u-invar'),
                            mixer={'backend': 'no-mixing'},
                            nbands='nao',
                            occupations={'name': 'fixed-uniform'})
            else:
                calc = GPAW(**calc_args,
                            txt=name + '_scf.txt',
                            occupations=FermiDirac(width=0.0, fixmagmom=True))
            atoms.calc = calc

            try:
                e, iters, t = tools_and_data.get_energy_and_iters(atoms, dm)
                assert abs(saved_data[dm][name][0] - e) < 1.0e-2
                assert abs(saved_data[dm][name][1] - iters) < 3
                print(name + "\t{}".format(iters),
                      file=fd, flush=True)

            except ConvergenceError:
                print(name + "\t{}".format(None),
                      file=fd, flush=True)
