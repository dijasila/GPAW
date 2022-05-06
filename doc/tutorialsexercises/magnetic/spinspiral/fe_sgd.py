from ase.build import bulk
from gpaw.new.ase_interface import GPAW

# Construct bulk iron with FCC lattice
atoms = bulk('Fe', 'fcc', a=3.50)  # V = 10.72
# atoms = bulk('Fe', 'fcc', a=3.577)  # V = 11.44

# Align the magnetic moment in the xy-plane
magmoms = [[1, 0, 0]]
ecut = 600
k = 14

# Construct list of q-vectors
path = atoms.cell.bandpath('GXW', npoints=31)

results = []
for i, q_c in enumerate(path.kpts):
    # Spin-spiral calculations require non-collinear calculations
    # without symmetry or spin-orbit coupling
    calc = GPAW(mode={'name': 'pw',
                      'ecut': ecut,
                      'qspiral': q_c},
                xc='LDA',
                symmetry='off',
                magmoms=magmoms,
                kpts=(k, k, k),
                txt=f'gs-{i:02}.txt')
    atoms.calc = calc
    atoms.get_potential_energy()
