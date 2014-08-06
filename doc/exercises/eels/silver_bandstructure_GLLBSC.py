import numpy as np
from ase.lattice import bulk
from gpaw import GPAW

# Part 1: Ground state calculation
atoms = bulk('Ag', 'fcc', a=4.090)      # Generate fcc crystal structure for silver, just use experimental lattice constant. 

# Ground state calculation:
calc = GPAW(mode='pw',            
            xc = 'GLLBSC',
            kpts = (10,10,10))

atoms.set_calculator(calc)
atoms.get_potential_energy()          
calc.write('Ag_GLLBSC.gpw')    


# Part 2: Bandstructure calculation
from ase.dft.kpoints import ibz_points, get_bandpath
points = ibz_points['fcc']
G = points['Gamma']
X = points['X']
W = points['W']
K = points['K']
L = points['L']
kpts, x, X = get_bandpath([W, L, G, X, W, K], atoms.cell)  # k-points as a path between high symmetry points in the Brillouin Zone


calc = GPAW('Ag_GLLBSC.gpw',        # load previos calculation 
            kpts=kpts,           # set new k-points
            fixdensity=True,     # fix density
            usesymm=None)        # Don't use symmetry to reduce number of k-points. 

calc.diagonalize_full_hamiltonian(nbands=20) # Diagonalize Hamiltonian, converge all bands
calc.write('Ag_bands_GLLBSC.gpw')

# Part 3: plot bandstructure
import pylab as p

calc = GPAW('Ag_bands_GLLBSC.gpw', txt=None)
nbands = calc.get_number_of_bands()
kpts = calc.get_ibz_k_points()
nkpts = len(kpts)

eigs = np.empty((nbands, nkpts), float)

# Get energies of all bands
for k in range(nkpts):
    eigs[:, k] = calc.get_eigenvalues(kpt=k)

# Subtract Fermi level from the self-consistent calculation
eigs -= GPAW('Ag_GLLBSC.gpw', txt=None).get_fermi_level()
for n in range(nbands):
    p.plot(eigs[n], 'm')
    
p.plot((0,50),(0,0), '--k')
p.xticks([0,10,22,37,44,49],['W','L', 'G', 'X', 'W', 'K' ])
p.xlabel('k-point')
p.ylabel('E-E${}_F$')
p.ylim(-7,7)
p.show()








