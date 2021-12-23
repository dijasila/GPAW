# Import modules
from gpaw import GPAW, PW, FermiDirac
from gpaw.test import equal
from ase.build import bulk
import numpy as np
from My_classes.Exchange_calculator import IsotropicExchangeCalculator, \
    compute_magnon_energy_simple, compute_magnon_energy_FM


def test_Fe_bcc():
    a = 2.867
    mm = 2.21
    atoms = bulk('Fe', 'bcc', a=a)
    atoms.set_initial_magnetic_moments([mm])

    # Calculation settings
    k = 4
    pw = 200
    xc = 'LDA'
    nbands_gs = 10   # Number of bands in ground state calculation
    # Number of bands to converge and use for response calculation
    nbands_response = 8
    conv = {'density': 1e-8,
            'forces': 1e-8,
            'bands': nbands_response}
    ecut = 50
    sitePos_mv = atoms.positions
    shapes_m = 'sphere'
    # All high symmetry points of the bcc lattice
    q_qc = np.array([[0, 0, 0],           # Gamma
                     [0.5, -0.5, 0.5],    # H
                     [0.0, 0.0, 0.5],     # N
                     [0.25, 0.25, 0.25]   # P
                     ])

    # Construct calculator
    calc = GPAW(xc=xc,
                mode=PW(pw),
                kpts={'size': (k, k, k), 'gamma': True},
                nbands=nbands_gs,
                occupations=FermiDirac(0.01),
                symmetry={'point_group': False},
                idiotproof=False,
                parallel={'domain': 1},
                spinpol=True,
                convergence=conv
                )

    # Do ground state calculation
    atoms.set_calculator(calc)
    atoms.get_potential_energy()
