from ase import Atoms
from gpaw import Calculator
from gpaw.dscf import dscf_calculation, MolecularOrbital, AEOrbital
from gpaw.utilities import equal

atoms = Atoms(positions=[[0.,0.,5.],
                         [0.,0.,6.1],
                         [0.,0.,3.]],
              symbols='H2Al',
              cell=[3.,3.,9.1],
              pbc=[True,True,False])

calc = Calculator(h=0.24,
                  nbands=6,
                  xc='PBE',
                  spinpol = True,
                  kpts=[1,1,1],
                  width=0.1,
                  convergence={'energy': 0.01,
                               'density': 1.0e-2,
                               'eigenstates': 1.0e-6,
                               'bands': -1})

atoms.set_calculator(calc)
e_gs = atoms.get_potential_energy()

## lumo = MolecularOrbital(calc, molecule=[0,1],
##                          w=[[1.,0.,0.,0.],[-1.,0.,0.,0.]])
## dscf_calculation(calc, [[1.0, lumo, 1]], atoms)
## e_1 = atoms.get_potential_energy()

H2 = atoms.copy()
del H2[-1]
calc2 = Calculator(h=0.24,
                   #nbands=6,
                   xc='PBE',
                   spinpol = True,
                   kpts=[1,1,1],
                   width=0.1,
                   convergence={'energy': 0.01,
                                'density': 1.0e-2,
                                'eigenstates': 1.0e-6,
                                'bands': -1})
H2.set_calculator(calc2)
H2.get_potential_energy()
wf_u = [kpt.psit_nG[1] for kpt in calc2.wfs.kpt_u]
P_aui = [[kpt.P_ani[a][1] for kpt in calc2.wfs.kpt_u]
          for a in range(len(H2))]

lumo = AEOrbital(calc, wf_u, P_aui, molecule=[0,1])
dscf_calculation(calc, [[1.0, lumo, 1]], atoms)
e_2 = atoms.get_potential_energy()

equal(e_2, e_gs + 3.0, 0.01)

del lumo
del calc.occupations
del calc
