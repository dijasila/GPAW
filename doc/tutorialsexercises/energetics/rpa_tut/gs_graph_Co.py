from pathlib import Path

from ase import Atoms
from ase.build import hcp0001
from gpaw import GPAW, PW, FermiDirac
from gpaw.hybrids.energy import non_self_consistent_energy as nsc_energy
from gpaw.mpi import world
from gpaw.xc.rpa import RPACorrelation

# Lattice parametes of Co:
a = 2.51
c = 4.07
slab = hcp0001('Co', a=a, c=c, size=(1, 1, 4), vacuum=10.0)
m = 1.3
slab.set_initial_magnetic_moments([m, m, m, m])

# Add graphite (we adjust height later):
slab += Atoms('C2',
              scaled_positions=[[0, 0, 0],
                                [1 / 3, 1 / 3, 0]],
              cell=slab.cell)


def calculate(d: float) -> None:
    slab.positions[4:6, 2] = slab.positions[3, 2] + d
    tag = f'{d:.3f}'
    slab.calc = GPAW(xc='PBE',
                     mode=PW(600),
                     nbands='200%',
                     kpts={'size': (16, 16, 1), 'gamma': True},
                     occupations=FermiDirac(width=0.01),
                     txt=f'gs-{tag}.txt')
    slab.get_potential_energy()
    E_hf = nsc_energy(slab.calc, 'EXX').sum()

    slab.calc.diagonalize_full_hamiltonian()
    slab.calc.write(f'tmp-{tag}.gpw', mode='all')

    rpa = RPACorrelation(f'tmp-{tag}.gpw',
                         txt=f'rpa-{tag}.txt',
                         skip_gamma=True,
                         frequency_scale=2.5)
    E_rpa = rpa.calculate(ecut=[200])[0]

    if world.rank == 0:
        Path(f'tmp-{tag}.gpw').unlink()
        with open(f'result-{tag}.out', 'w') as fd:
            print(d, E_hf, E_rpa, file=fd)


if __name__ == '__main__':
    import sys
    d = float(sys.argv[1])
    calculate(d)
