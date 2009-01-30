from ase import *
from gpaw import *
from gpaw import dscf

# Ground state calculation
#------------------------------------------------------------------

calc = GPAW(nbands=8, h=0.2, xc='PBE', spinpol=True,
            convergence={'energy': 100,
                         'density': 100,
                         'eigenstates': 1.0e-9,
                         'bands': -1})

CO = molecule('CO')
CO.center(vacuum=3)
CO.set_calculator(calc)

E_gs = CO.get_potential_energy()

calc.write('CO.gpw', mode='all')
CO, calc = restart('CO.gpw')

## '''Obtain the pseudowavefunctions and projector overlaps of the
##  state which is to be occupied. n=5,6 is the 2pix and 2piy orbitals'''
wf_u = [kpt.psit_nG[5] for kpt in calc.wfs.kpt_u]
P_aui = [[kpt.P_ani[a][5] for kpt in calc.wfs.kpt_u]
          for a in range(len(CO))]

# Excited state calculation
#--------------------------------------------

lumo = dscf.AEOrbital(calc, wf_u, P_aui, molecule=[0,1])
#lumo = dscf.MolecularOrbital(calc, molecule=[0,1], w=[[0,0,0,1],[0,0,0,-1]])
dscf.dscf_calculation(calc, [[1.0, lumo, 1]], CO)

E_es = CO.get_potential_energy()

print 'Excitation energy: ', E_es-E_gs
