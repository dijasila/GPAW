from ase.build import molecule
from gpaw import GPAW
from gpaw.utilities import h2gpts
from gpaw.poisson_image import ImagePoissonSolver as EPS2
from ase.parallel import parprint

atoms = molecule('NH3')
atoms.numbers[0] = 8
atoms.center(vacuum=5.0)

atoms.pbc = (1,1,0)
gpts = h2gpts(0.2, atoms.cell, idiv=16)
extgpts = (gpts[0], gpts[1], gpts[2] * 2)

calc = GPAW(mode='lcao',
            gpts=gpts,
            charge=1,
            basis='dzp',
            poissonsolver=EPS2(direction=2, side='left',
                               extended={'gpts': extgpts, 'useprev': True}),
            txt='gpaw_h3o+.txt')

atoms.calc = calc
atoms.get_potential_energy()
pot = calc.get_electrostatic_potential().mean(0).mean(0) 

parprint('Potential[1]', pot[1])
parprint('Potential[-1]', pot[-1])

assert abs(pot[1] - -0.220660556026) < 1e-3
assert abs(pot[-1] - -6.61986899999) < 1e-3
# assert abs(pot[1] - -0.04969663) < 1e-3
# assert abs(pot[-1] - -6.56813735) < 1e-3

