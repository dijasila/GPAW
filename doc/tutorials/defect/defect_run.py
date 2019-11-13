import numpy as np

from ase.io import read

from gpaw import GPAW, FermiDirac, Mixer

from disorder import Disorder

PP = 'PAW'
mode = 'FD' # 'LCAO' # 
basis = 'SZ'
h = 0.15
k = 3
kpts = (k, k, 1)
N = 11
sigma = 5.0e-3

Ne = 8

defect = 'C1'

if defect=='C1':
    vac = [(0, 0), ]
else:
    vac = [(0, 1), ]

convergence = {'energy': 5e-4/(Ne*N**2),
               'bands': Ne/2*N**2 + 20}
parallel = {'domain': (3, 3, 2),
            # 'sl_default': (4, 4, 64),
            # 'band': 2,
            }

calc = GPAW(parallel=parallel,
            convergence=convergence,
            kpts=kpts,
            maxiter=500,
            mode=mode.lower(),
            basis='%s(dzp)' % basis.lower(),
            symmetry={'point_group': True,
                      'tolerance': 1e-6},
            xc='LDA',
            mixer=Mixer(beta=0.05, nmaxold=5, weight=100.0),
            nbands=Ne*N**2/2+70,
            setups={'default': PP.lower()},
            occupations=FermiDirac(sigma),
            h=h,
            # gpts=gpts[h],
            txt='vacancy_%s_pbc_%s_%s_%s_%s_k_%u_h_%.1e_N_%u.txt' % \
                (defect, pbc, PP, mode, basis, k, h, N),
            verbose=1)
 
# Atoms
atoms = read('graphene_a_2.46.traj')
atoms.set_pbc([1, 1, 0])

fname = 'vacancy_%s_pbc_%s_%s_%s_%s_k_%u_h_%1.1e_N_%u' \
        % (defect, pbc, PP, mode, basis, k, h, N)
disorder = Disorder(atoms, calc=calc, supercell=(N, N, 1), defect=vac,
                    name=fname, pckldir='./pickle')
disorder.run()

calc.write('gpw/calc_vacancy_%s_pbc_%s_%s_%s_%s_k_%u_h_%1.1e_N_%u.defect.gpw' \
               % (defect, pbc, PP, mode, basis, k, h, N))
