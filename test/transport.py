from ase import *
from gpaw import *
from gpaw.lcao.gpawtransport import GPAWTransport 
from gpaw.atom.basis import BasisMaker
import pickle

a = 3.6
L = 7.00 

basis = BasisMaker('Na').generate(1, 1)

atoms = Atoms('Na4', pbc=(1, 1, 1), cell=[4 * a, L, L])
atoms.positions[:4, 0] = [i * a for i in range(4)]
atoms.positions[:, 1:] = L / 2.
atoms.center()
atoms.set_calculator(GPAW(h=0.3,
                          xc='PBE',
                          basis={'Na': basis},
                          kpts=(3,1,1),
                          width=0.01,
                          mode='lcao',
                          txt='Na_lcao.txt',
                          usesymm=None,
                          mixer=Mixer(0.1, 5, metric='new', weight=100.0)))
pl_atoms1 = range(4)     
pl_atoms2 = range(-4, 0) 
pl_cell1 = (4 * a, L, L) 
pl_cell2 = pl_cell1      

gpawtran = GPAWTransport(atoms=atoms,
                         pl_atoms=(pl_atoms1, pl_atoms2),
                         pl_cells=(pl_cell1, pl_cell2),
                         d=0) 
gpawtran.negf_prepare('test1', scat_restart=False, lead_restart=False)
gpawtran.get_selfconsistent_hamiltonian(bias=0, gate=0,extend_layer=0,verbose=1)
filename = 'Na4_eq'
fd = file(filename,'wb')
pickle.dump((gpawtran.h_syzkmm, gpawtran.d_syzkmm, gpawtran.atoms.calc.hamiltonian.vt_sG), fd, 2)
fd.close()

