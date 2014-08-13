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
    setups={'Ni': '10'},
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

# Setup 6eV Hubbard U on the 3d-orbitals (l=2) of Ni atoms:
calc.set(setups={'Ni': '10:3d,6.0'})
## Make ready for scf with the DFT+U functional and converge this new system
## and get new band gap.....which should be much larger:
e2 = calc.get_potential_energy()
Eg_Hub = band_gab(calc)


## Also test that new interface for setting U works
U_ev = 6.0              # U in eV
U_au = U_ev / Hartree   # U in atomic units

# Setting on atom 0,1 : l=2, for spins (no spin dependency) 
# This way allows setting U values individually for all
# atoms and orbitals.
# {atom index:{n:{l:{'U': U_value [in Hartree]}}}}
HubU_dict = {0:{3:{2:{'U':U_au}}},
             1:{3:{2:{'U':U_au}}},
            'scale': 1,
            }

calc.hamiltonian.set_hubbard_u(HubU_dict=HubU_dict)
calc.scf.reset()
e2_new_interface = calc.get_potential_energy()
Eg_Hub_new_interface = band_gab(calc)
# This should be the same as Eg_Hub (any difference due to numerics) 
# (perhaps wasteful way of just testing the interface)

equal(Eg_Hub, 4.7, 0.2)
equal(Eg_non_Hub, 0.8, 0.1)
equal(Eg_Hub_new_interface, Eg_Hub, 1e-3)  

