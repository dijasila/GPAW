import numpy as np
from ase.io import read, write, iread
from ase.parallel import parprint, paropen
from gpaw import GPAW, FD, FermiDirac, PW, restart, LCAO, ConvergenceError
from copy import deepcopy
from gpaw.mpi import world, rank, MASTER
import sys
np.set_printoptions(threshold = sys.maxsize)

calc = GPAW(xc='PBE',
                mode=LCAO(),
                h=0.2,
                basis='dzp',
                spinpol=True,
                eigensolver='etdm',
                occupations={'name': 'fixed-uniform'},
                mixer={'backend': 'no-mixing'},
                nbands='nao',
                symmetry='off',
                txt='N-Phenylpyrrole_GS.txt')

atoms = read('N-Phenylpyrrole.xyz')
atoms.center(vacuum = 5.0)
atoms.set_pbc(False)
atoms.calc = calc

# Ground state calculation
E_GS = atoms.get_potential_energy()

# Ground state LCAO coefficients and occupation numbers
C_GS = [calc.wfs.kpt_u[x].C_nM.copy() for x in range(len(calc.wfs.kpt_u))]
f_gs = [calc.wfs.kpt_u[x].f_n.copy() for x in range(len(calc.wfs.kpt_u))]

# Direct approach using ground state orbitals with changed occupation numbers
calc.set(eigensolver=ETDM(searchdir_algo={'name': 'l-sr1p'},
                          linesearch_algo={'name': 'max-step'},
                          need_init_orbs=False),
         txt='N-Phenylpyrrole_EX_direct.txt')

# Spin-mixed open-shell occupation numbers
f = excite(calc, 0, 0, spin=(0, 0))

atoms, calc0 = restart('/users/work/yorick/PP/pp_gs.gpw', txt = 'gpaw0.txt')
calc0.calculate(properties = ['energy'], system_changes = ['positions'])
h = 26
p = 27
occ = calc0.wfs.kpt_u[0].f_n.copy()
occ_ex_up = occ.copy()
occ_ex_down = occ.copy()
occ_ex_up[h] = 0
occ_ex_up[p] = 1
occ = [occ_ex_up, occ_ex_down]

calc = GPAW(xc = 'PBE',
            basis = 'cc-pVDZ_PBE.sz',
            mode = LCAO(force_complex_dtype = False),
            h = 0.15,
            spinpol = True,
            nbands = "nao",
            eigensolver = {'name': 'etdm',
                           'representation': 'u-invar',
                           'partial_diagonalizer': {'name': 'Davidson',
                                                    'logfile': 'davidson.txt',
                                                    'remember_sp_order': True,
                                                    'sp_order': 6,
                                                    'm': np.inf,
                                                    'h': 1e-3,
                                                    'eps': 1e-2},
                           'linesearch_algo': {'name': 'maxstep', 'max_step': 0.2},
                           'searchdir_algo': {'name': 'LBFGS-P_MMF'},
                           'update_ref_orbs_counter': np.inf,
                           'need_init_orbs': False},
            occupations = {'name': 'mom', 'numbers': occ, 'use_fixed_occupations': True},
            mixer = {'backend': 'no-mixing'},
            convergence = {'energy': 1e-7},
            txt = 'gpaw.txt',
            symmetry = 'off',
            parallel = {'domain': world.size})

calc.initialize(atoms)
C = [deepcopy(calc0.wfs.kpt_u[x].C_nM) for x in range(len(calc0.wfs.kpt_u))]
eps = [deepcopy(calc0.wfs.kpt_u[x].eps_n) for x in range(len(calc0.wfs.kpt_u))]
atoms.set_calculator(calc)
for k in range(len(C)):
    calc.wfs.kpt_u[k].C_nM = deepcopy(C[k])
    calc.wfs.kpt_u[k].eps_n = deepcopy(eps[k])
calc.wfs.eigensolver.c_nm_ref = deepcopy(C)
atoms.get_potential_energy()