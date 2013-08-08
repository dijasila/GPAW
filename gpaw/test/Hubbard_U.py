# -*- coding: utf-8 -*-
import numpy as np
from math import sqrt
from ase.units import Hartree
from ase import Atoms
from gpaw import GPAW, FermiDirac, PoissonSolver
from gpaw.test import equal

##############################################################################
## Define a function that returns the band gab
## defined as the difference between the bottom of the conduction band and the
## top of the valence band.
def band_gab(calc):
    ef = calc.get_fermi_level()
    Nb = calc.wfs.bd.nbands
    w_k = calc.wfs.weight_k
    x = 0
    nspin=calc.get_number_of_spins()
    energies = np.empty(len(w_k) * Nb*nspin)
    for spin in np.arange(nspin):
        for k, w in enumerate(w_k):
            energies[x:x + Nb] = calc.get_eigenvalues(k,spin)
            x += Nb
    index1=np.where(energies-ef<=0)
    index2=np.where(energies-ef>0)

    Vb=max(energies[index1[0]])-ef
    Cb=min(energies[index2[0]])-ef
    return Cb-Vb





##############################################################################
## Setup up bulk NiO in an antiferromagnetic configuration
name='Hubbard_test_on_NiO'
a = 4.19 # Lattice constants
b=a/sqrt(2)
m=2
k=2 # Number of k-points
atoms = Atoms(symbols='Ni2O2',
              pbc=True,
              cell=(b, b, a),
              positions=[(0, 0, 0),
                         (b/2, b/2, a/2),
                         (0, 0, a/2),
                         (b/2, b/2, 0)],
              magmoms=(m,-m,0,0)
              )

##############################################################################
## Setup the calculator
calc = GPAW(
    h=0.25,
    occupations=FermiDirac(width=0.05),
    poissonsolver=PoissonSolver(nn='M', relax='J'),
    convergence={'eigenstates':8e-4,'density': 1.0e-2,'energy': 0.1},
    #txt=name+'.txt',
    kpts=(k, k, k),
    xc='PBE')

atoms.set_pbc(1)
atoms.set_calculator(calc)

##############################################################################
## Find the  ground-state and get the band gab
e1 = atoms.get_potential_energy()
Eg_non_Hub = band_gab(calc)

# Setup 6eV Hubbard U on the d-orbitals (l=2) of Ni atoms:
calc.set(setups={'Ni': '10:d,6.0'})
## Make ready for scf with the DFT+U functional and converge this new system
## and get new band bag.....which should be much larger:
e2 = calc.get_potential_energy()
Eg_Hub = band_gab(calc)

##############################################################################
## Now we expect that one effect of the Hubbard U is the opening of the band
## gab, so the band gab shall we test parameter:
## Let's compare the new and old band gab and require that is has opened by
## at least 0.2 eV
#assert( Eg_Hub- Eg_non_Hub>1.9)
print(Eg_Hub, Eg_non_Hub)
equal(Eg_Hub, 4.5, 0.3)
equal(Eg_non_Hub, 0.8, 0.1)
