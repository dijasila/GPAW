import os
from ase import *
from ase.parallel import rank, barrier
from gpaw.test import equal
from gpaw import GPAW
from gpaw.atom.generator import Generator, parameters
from gpaw.xc_functional import XCFunctional
from gpaw import setup_paths

# Generate setup
symbol = 'Li'
if rank == 0:
    g = Generator(symbol, 'revPBE', scalarrel=True, nofiles=True)
    g.run(exx=True, **parameters[symbol])
barrier()
setup_paths.insert(0, '.')

a = 5.0
n = 24
li = Atoms(symbol, magmoms=[1.0], cell=(a, a, a), pbc=True)

calc = GPAW(gpts=(n, n, n), nbands=1, xc='PBE')
li.set_calculator(calc)
e = li.get_potential_energy() + calc.get_reference_energy()
equal(e, -7.462 * Hartree, 1.4)

calc.set(xc='revPBE')
erev = li.get_potential_energy() + calc.get_reference_energy()

del setup_paths[0]

equal(erev, -7.487 * Hartree, 1.3)
equal(e - erev, 0.025 * Hartree, 0.002 * Hartree)
