from gpaw import GPAW, restart
from ase.structure import molecule
from gpaw.test import equal
esolvers = ['cg', 'rmm-diis', 'dav']

calc = GPAW(xc='LDA',
            eigensolver='cg',
            convergence={'eigenstates': 3.5e-5},
            #txt=None,
            dtype=complex)

mol = molecule('N2')
mol.center(vacuum=3.0)
mol.set_calculator(calc)

Eini = mol.get_potential_energy()

calc.write('N2.gpw', mode='all')
del calc, mol

E = {}
for esolver in esolvers:

    mol, calc = restart('N2.gpw', txt=None)

    if (calc.wfs.dtype!=complex or
        calc.wfs.kpt_u[0].psit_nG.dtype!=complex):
        raise AssertionError('ERROR: restart failed to read complex WFS')
    
    calc.scf.reset()
    calc.set(convergence={'eigenstates': 3.5e-9})
    calc.set(eigensolver=esolver)

    E[esolver] = mol.get_potential_energy()
    
    print(esolver, E[esolver])

for esolver in esolvers:
    print esolver
    equal(E[esolver], Eini, 1E-8)
