from ase import *
from gpaw import *
from gpaw.lcao.gpawtransport import GPAWTransport 
from gpaw.atom.basis import BasisMaker
import pickle
import numpy as npy

a = 3.6 # Na binding length
L = 7.00 # width of unit cell

# Setup the Atoms for the scattering region.
atoms = Atoms('Na12', pbc=(1, 1, 1), cell=[12 * a, L, L])
atoms.positions[:5, 0] = [i * a for i in range(5)]
atoms.positions[-5:, 0] = [i * a + 3 * a for i in range(4, 9)]
atoms.positions[5:7, 0] = [5 * a, 6 * a ]
atoms.positions[:, 1:] = L / 2.

bm = BasisMaker('Na')#, run=False)
#bm.generator.run(write_xml=False)
basis = bm.generate(1, 1)

atoms.center()
# Attach a GPAW calculator, attention to two keywords: fortransport, usesymm
atoms.set_calculator(GPAW(h=0.3,
                          xc='PBE',
                          basis={'Na': basis},
                          kpts=(3,1,1),
                          width=0.01,
                          mode='lcao',
                          txt='Na_lcao.txt',
                          usesymm=None,
                          mixer=Mixer(0.1, 5, metric='new', weight=100.0)))

# Setup the GPAWTransport calculator
pl_atoms1 = range(4)     # Atomic indices of the left principal layer
pl_atoms2 = range(-4, 0) # Atomic indices of the right principal layer
pl_cell1 = (4 * a, L, L) # Cell for the left principal layer
pl_cell2 = pl_cell1      # Cell for the right principal layer

gpawtran = GPAWTransport(atoms=atoms,
                         pl_atoms=(pl_atoms1, pl_atoms2),
                         pl_cells=(pl_cell1, pl_cell2),
                         d=0) #transport direction (0 := x)

#generte the guess for transport selfconsistent calculation
#flag: 0 for calculation, 1 for restart
gpawtran.negf_prepare('test1', 0)

#the result restored in the h_syzkmm member of gpawtransport object
gpawtran.get_selfconsistent_hamiltonian(bias=0, gate=0, verbose=1)

filename = 'Na12_eq'
fd = file(filename,'wb')
pickle.dump((gpawtran.h_syzkmm, gpawtran.d_syzkmm, gpawtran.atoms.calc.hamiltonian.vt_sG), fd, 2)
fd.close()
