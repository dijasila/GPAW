from pathlib import Path

import numpy as np
from ase.build import add_adsorbate, hcp0001
from ase.dft.kpoints import monkhorst_pack
from ase.parallel import paropen
from gpaw import GPAW, PW, FermiDirac
from gpaw.hybrids.energy import non_self_consistent_energy as nsc_energy
from gpaw.mpi import rank
from gpaw.xc.rpa import RPACorrelation

kpts = monkhorst_pack((16, 16, 1))
kpts += np.array([1 / 32, 1 / 32, 0])

a = 2.51  # lattice parameter of Co
slab = hcp0001('Co', a=a, c=4.07, size=(1, 1, 4))
pos = slab.get_positions()
cell = slab.get_cell()
cell[2, 2] = 20. + pos[-1, 2]
slab.set_cell(cell)
slab.set_initial_magnetic_moments([0.7, 0.7, 0.7, 0.7])

ds = [1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 5.0, 6.0, 10.0]

for d in ds:
    pos = slab.get_positions()
    add_adsorbate(slab, 'C', d, position=(pos[3, 0], pos[3, 1]))
    add_adsorbate(slab, 'C', d, position=(cell[0, 0] / 3 + cell[1, 0] / 3,
                                          cell[0, 1] / 3 + cell[1, 1] / 3))
    # view(slab)
    calc = GPAW(xc='PBE',
                mode=PW(600),
                kpts=kpts,
                occupations=FermiDirac(width=0.01),
                txt=f'hmm2_gs_{d}.txt')
    slab.calc = calc
    E = slab.get_potential_energy()
    E_hf = nsc_energy(calc, 'EXX')

    calc.diagonalize_full_hamiltonian()
    calc.write(f'gs_{d}.gpw', mode='all')

    f = paropen('hf_acdf.dat', 'a')
    print(d, E, E_hf, file=f)
    f.close()

    del slab[-2:]

    ecut = 200

    rpa = RPACorrelation(f'gs_{d}.gpw', txt=f'rpa_{ecut}_{d}.txt')
    E_rpa = rpa.calculate(ecut=[ecut],
                          frequency_scale=2.5,
                          skip_gamma=True,
                          filename=f'restart_{ecut}_{d}.txt')

    f = paropen(f'rpa_{ecut}.dat', 'a')
    print(d, E_rpa, file=f)
    f.close()

    if rank == 0:
        Path(f'gs_{d}.gpw').unlink()
