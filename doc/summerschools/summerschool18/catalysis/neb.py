import numpy as np
from ase.io import read
from ase.constraints import FixAtoms
from ase.neb import NEB
from ase.optimize import BFGS
from ase.io.trajectory import Trajectory
from ase.parallel import rank, size
from gpaw import GPAW, PW

initial = read('ini.POSCAR')
final = read('fin.POSCAR', -1)

images = [initial]
N = 4  #No. of images

z = initial.positions[:,2]
constraint = FixAtoms(mask=(z < z.min() + 1.0))

j = rank*N//size
n = size // N  # number of cpu's per image

for i in np.r_[0:N]:
    image = initial.copy()
    image.set_constraint(constraint)
    if i ==j:
        calc = GPAW(xc='PBE', mode=PW(350), communicator = range(j*2,j*2+n), txt = '%i.txt'%i, kpts={'size':(4,4,1),'gamma':True}, convergence={'eigenstates':1e-7})
        image.set_calculator(calc)
    images.append(image)
images.append(final)

neb = NEB(images, k=0.5, parallel=True)
neb.interpolate()
qn = BFGS(neb)
if rank % (size // N) == 0:
    traj = Trajectory('neb%d.traj' % j, 'w', images[1 + j], master=True)
    qn.attach(traj)
qn.run(fmax=0.05)
