from ase import Atoms
from ase.parallel import paropen

from gpaw import GPAW, FermiDirac

# Script to get the total energies of a supercell
# of GaAs with and without a Ga vacancy

a = 5.628 # Lattice parameter
N = 2 #  NxNxN supercell
q = -3 # Defect charge

formula  = 'Ga4As4'

lattice = [[a, 0.0 ,0.0], # work with cubic cell
           [0.0,a,0.0],
           [0.0,0.0,a]]

basis = [[0.0,0.0,0.0],
         [0.5,0.5,0.0],
         [0.0,0.5,0.5],
         [0.5,0.0,0.5],
         [0.25,0.25,0.25],
         [0.75,0.75,0.25],
         [0.25,0.75,0.75],
         [0.75,0.25,0.75]]

GaAs = Atoms(symbols=formula,
             scaled_positions=basis,
             cell=lattice,
             pbc=(1,1,1)
            )

GaAsdef = GaAs.repeat((N,N,N))
GaAsdef.pop(0) # Make the supercell and a Ga vacancy

calc = GPAW(mode='fd',
            kpts={'size': (2,2,2),'gamma':False},
            xc='LDA',
            charge=q,
            occupations=FermiDirac(0.01),
            txt='GaAs.Ga_vac.txt')

GaAsdef.set_calculator(calc)
Edef = GaAsdef.get_potential_energy()

calc.write('GaAs.Ga_vac.gpw')

# Now for the pristine case

GaAspris = GaAs.repeat((N,N,N))

calc = GPAW(mode='fd',
            kpts={'size': (2,2,2),'gamma':False},
            xc='LDA',
            occupations=FermiDirac(0.01),
            txt='GaAs.pristine.txt')

GaAspris.set_calculator(calc)
Epris = GaAspris.get_potential_energy()

calc.write('GaAs.pristine.gpw')

outfile = paropen('results.dat','w')
s = '# cell_size defective_energy pristine_energy difference \n'
outfile.write(s)
s = str(N) + ' ' + str(Edef) + ' ' + str(Epris) + ' ' + str(Edef - Epris) +  '\n'
outfile.write(s)
