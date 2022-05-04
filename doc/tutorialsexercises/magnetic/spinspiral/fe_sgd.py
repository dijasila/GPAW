from ase.build import bulk
from gpaw import GPAW, PW, mpi
import numpy as np

# Construct bulk iron with FCC lattice
atoms = bulk('Fe', 'fcc', a=3.50)  # V = 10.72
# atoms = bulk('Fe', 'fcc', a=3.577)  # V = 11.44

# Align the magnetic moment in the xy-plane
magmoms = np.array([[1, 0, 0]])
pw = 1000
k = 18

# Construct list of q-vectors
path = atoms.cell.bandpath('GXW', npoints=31)
Q = path.kpts

e = []
mT = []
for i, q_c in enumerate(Q):
    # Spin-spiral calculations require non-collinear calculations
    # without symmetry or spin-orbit coupling
    calc = GPAW(mode=PW(pw, qspiral=q_c),
                xc='LDA',
                symmetry='off',
                magmoms=magmoms,
                kpts=(k, k, k),
                txt=f'{i}gs.txt')

    atoms.set_calculator(calc)
    e.append(atoms.get_potential_energy())
    totmom_v, magmom_av = calc.density.estimate_magnetic_moments()
    mT.append(totmom_v)

e = np.array(e)
mT = np.array(mT)

# Save result in .npy file
if mpi.world.rank == 0:
    with open(f'spiral_{pw}pw{k}k.npy', 'wb') as f:
        np.save(f, [path])
        np.save(f, e)
        np.save(f, mT)
