from __future__ import print_function
from gpaw import GPAW
from ase.structure import molecule
from gpaw.mpi import world

system = molecule('H2O')
system.center(vacuum=1.2)
system.pbc = 1

band = 1
if world.size > 4:
    band = 2

for mode in [#'fd',
             #'pw',
             'lcao'
             ]:
    energy = []
    for augment_grids in [True, False]:
        if mode != 'lcao':
            eigensolver = 'rmm-diis'
        else:
            eigensolver = None
        calc = GPAW(mode=mode,
                    eigensolver=eigensolver,
                    parallel=dict(augment_grids=augment_grids,
                                  band=band),
                    basis='szp(dzp)',
                    kpts=[1, 1, 4],
                    nbands=8)
        def stopcalc():
            calc.scf.converged = True
        # Iterate enough for density to update so it depends on potential
        calc.attach(stopcalc, 5)
        system.set_calculator(calc)
        energy.append(system.get_potential_energy())
    err = energy[1] - energy[0]
    assert err < 1e-10, err
