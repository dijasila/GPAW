"""Test the calclation of the excitation energy of Na2 by RSF and IVOs."""
from ase.build import molecule
from gpaw import GPAW, setup_paths
from gpaw.occupations import FermiDirac
from gpaw.test import equal, gen
from gpaw.eigensolvers import RMMDIIS
from gpaw.cluster import Cluster

h = 0.35  # Grispacing
e_singlet = 1.82  # eV by NIST

if setup_paths[0] != '.':
    setup_paths.insert(0, '.')

gen('Na', xcname='PBE', scalarrel=True, exx=True, yukawa_gamma=0.38)

c = {'energy': 0.001, 'eigenstates': 3, 'density': 3}
na2 = Cluster(molecule('Na2'))
na2.minimal_box(4, h=h)
calc = GPAW(txt='na2_ivo.txt', xc='LCY_PBE:omega=0.38:excitation=singlet',
            eigensolver=RMMDIIS(), h=h, occupations=FermiDirac(width=0.0),
            spinpol=False, convergence=c)
na2.set_calculator(calc)
na2.get_potential_energy()
(eps_homo, eps_lumo) = calc.get_homo_lumo()
e_ex = eps_lumo - eps_homo
equal(e_singlet, e_ex, 0.15)
calc.write('na2.gpw')
c2 = GPAW('na2.gpw')
assert c2.hamiltonian.xc.excitation == 'singlet'
