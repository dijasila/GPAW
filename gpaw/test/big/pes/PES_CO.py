# Takes 30 mins with 8 cores in domain-parallel (memory ~500mb, max 1.5gb)

import numpy as np
from ase import Atom, Atoms
from gpaw import GPAW, MixerDif, mpi
from gpaw.lrtddft import LrTDDFT
from gpaw.pes.dos import DOSPES
from gpaw.pes.tddft import TDDFTPES

atoms = Atoms(
    [Atom('C', (7.666263598984184, 7.637780850431168, 8.289450797111844)),
     Atom('O', (8.333644370007132, 8.362384430165646, 7.710230973847514))])
atoms.center(vacuum=8)

h = 0.15
N_c = np.round(atoms.get_cell().diagonal() / h / 16) * 16

m_calc = GPAW(gpts=N_c, nbands=6, mixer=MixerDif(0.1, 5, weight=100.0),
              # convergence={'bands':10},
              parallel={'domain': mpi.size},
              xc='PBE', txt='CO-m.txt', spinpol=True)

m = atoms.copy()
m.set_initial_magnetic_moments([-1, 1])
m.calc = m_calc
m.get_potential_energy()

d_calc = GPAW(gpts=N_c, nbands=16, mixer=MixerDif(0.1, 5, weight=100.0),
              convergence={'bands': 10},
              parallel={'domain': mpi.size},
              charge=1,
              xc='PBE', txt='CO-d.txt', spinpol=True)

d = atoms.copy()
d.set_initial_magnetic_moments([-1, 1])
d.calc = d_calc
d.get_potential_energy()

istart = 0  # band index of the first occ. band to consider
jend = 15   # band index of the last unocc. band to consider
d_lr = LrTDDFT(d_calc, xc='PBE', nspins=2,
               restrict={'istart': istart, 'jend': jend})
d_lr.diagonalize()

pes = TDDFTPES(m_calc, d_lr, d_calc)
pes.save_folded_pes('CO-td.dat', folding=None)

pes = DOSPES(m_calc, d_calc, shift=True)
pes.save_folded_pes('CO-dos.dat', folding=None)
