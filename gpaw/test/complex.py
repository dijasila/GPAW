from __future__ import print_function
from gpaw import GPAW, restart
from ase.build import molecule
from gpaw.test import equal
Eini0 = -17.8037610364
energy_eps = 5e-4
esolvers = ['cg', 'rmm-diis', 'dav']

calc = GPAW(xc='LDA',
            h=0.21,
            eigensolver='cg',
            convergence={'eigenstates': 3.5e-5, 'energy': energy_eps},
            txt=None,
            dtype=complex)

mol = molecule('N2')
mol.center(vacuum=3.0)
mol.set_calculator(calc)

Eini = mol.get_potential_energy()
Iini = calc.get_number_of_iterations()
print(('%10s: %12.6f eV in %3d iterations' %
       ('init(cg)', Eini, Iini)))
print(energy_eps * calc.get_number_of_electrons())
equal(Eini, Eini0, energy_eps * calc.get_number_of_electrons())

calc.write('N2_complex.gpw', mode='all')
del calc, mol

for esolver in esolvers:
    mol, calc = restart('N2_complex.gpw', txt=None)

    assert calc.wfs.dtype == complex
    assert calc.wfs.kpt_u[0].psit_nG.dtype == complex
    
    calc.scf.reset()
    calc.set(convergence={'eigenstates': 3.5e-9, 'energy': energy_eps})
    calc.set(eigensolver=esolver)
    E = mol.get_potential_energy()
    equal(E, Eini, energy_eps * calc.get_number_of_electrons())
