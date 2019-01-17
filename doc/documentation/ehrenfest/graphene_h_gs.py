import ase.io as io
import numpy as np

from ase.lattice.hexagonal import Graphene
from gpaw.mixer import Mixer, MixerSum
from gpaw import GPAW
from ase import Atom, Atoms
from gpaw.utilities import h2gpts
from ase.units import Bohr
from gpaw.occupations import FermiDirac
from gpaw.external import ConstantElectricField
from gpaw.mpi import world

def gaussian(x, x0, A):
   E = np.linalg.norm(x-x0)
   return A*np.exp(-E**2)

name = 'graphene_h'

# 5 x 5 supercell of graphene
index1 = 5
index2 = 5
a = 2.45
c = 3.355

gra = Graphene(symbol = 'C',latticeconstant={'a':a,'c':c},
               size=(index1,index2,1), pbc = (1,1,0))

gra.center(vacuum=15.0, axis=2)
gra.center()

# Starting position of the projectile with an impact point at the center of a hexagon
projpos = [[gra[15].position[0], gra[15].position[1]+1.41245, 25.0]]

H = Atoms('H', cell=gra.cell, positions=projpos)

# Combine target and projectile
atoms = gra + H
atoms.set_pbc(True)

conv_fast = {'energy':1.0, 'density': 1.0, 'eigenstates':1.0}
conv_par = {'energy':0.001, 'density': 1e-3, 'eigenstates':1e-7}
const_pot = ConstantPotential(1.0)
mixer= Mixer(0.1,5,weight=100.0)

calc = GPAW(gpts=h2gpts(0.2, gra.get_cell(), idiv=8),
            nbands = 110, xc='LDA',charge=1, txt=name + '_gs.txt',
            convergence=conv_fast, external=const_pot)

atoms.set_calculator(calc)
atoms.get_potential_energy()

A = 1.0
X0 = atoms.positions[50] / Bohr
rcut = 3.0 / Bohr
vext = calc.hamiltonian.vext
gd = calc.hamiltonian.finegd
n_c = gd.n_c
h_c = gd.get_grid_spacings()
b_c = gd.beg_c
vext.vext_g[:] = 0.0
for i in range(n_c[0]):
    for j in range(n_c[1]):
        for k in range(n_c[2]):
            x = h_c[0]*(b_c[0] + i)
            y = h_c[1]*(b_c[1] + j)
            z = h_c[2]*(b_c[2] + k)
            X = np.array([x,y,z])
            dist = np.linalg.norm(X-X0)
            if(dist < rcut):
                vext.vext_g[i,j,k] += gaussian(X,X0,A)

calc.set(convergence=conv_par, eigensolver=RMMDIIS(5), external=vext)

atoms.get_potential_energy()
calc.write(name + '.gpw', mode='all')
