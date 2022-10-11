from pathlib import Path

from ase import Atoms
from ase.build import fcc111
from gpaw import GPAW, PW, FermiDirac, MixerSum, Davidson
from gpaw.hybrids.energy import non_self_consistent_energy as nsc_energy
from gpaw.mpi import world
from gpaw.xc.rpa import RPACorrelation

# Lattice parametes of Cu:
d = 2.56
a = 2**0.5 * d
slab = fcc111('Cu', a=a, size=(1, 1, 4), vacuum=10.0)
slab.pbc = True

# Add graphite (we adjust height later):
slab += Atoms('C2',
              scaled_positions=[[0, 0, 0],
                                [1 / 3, 1 / 3, 0]],
              cell=slab.cell)


def calculate(xc: str, d: float) -> float:
    slab.positions[4:6, 2] = slab.positions[3, 2] + d
    tag = f'{d:.3f}'
    if xc == 'RPA':
        xc0 = 'PBE'
    else:
        xc0 = xc
    slab.calc = GPAW(xc=xc0,
                     mode=PW(800),
                     basis='dzp',
                     eigensolver=Davidson(niter=4),
                     nbands='200%',
                     kpts={'size': (12, 12, 1), 'gamma': True},
                     occupations=FermiDirac(width=0.05),
                     convergence={'density': 1e-5},
                     parallel={'domain': 1},
                     mixer=MixerSum(0.05, 5, 50),
                     txt=f'{xc0}-{tag}.txt')
    e = slab.get_potential_energy()

    if xc == 'RPA':
        e_hf = nsc_energy(slab.calc, 'EXX').sum()

        slab.calc.diagonalize_full_hamiltonian()
        slab.calc.write(f'{xc0}-{tag}.gpw', mode='all')

        rpa = RPACorrelation(f'{xc0}-{tag}.gpw',
                             txt=f'RPAc-{tag}.txt',
                             skip_gamma=True,
                             frequency_scale=2.5)
        e_rpac = rpa.calculate(ecut=[200])[0]
        e = e_hf + e_rpac

        if world.rank == 0:
            Path(f'{xc0}-{tag}.gpw').unlink()
            with open(f'RPAc-{tag}.result', 'w') as fd:
                print(d, e, e_hf, e_rpac, file=fd)

    return e


if __name__ == '__main__':
    import sys
    xc = sys.argv[1]
    for arg in sys.argv[2:]:
        d = float(arg)
        calculate(xc, d)
