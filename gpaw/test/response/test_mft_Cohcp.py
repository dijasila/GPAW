"""Calculate the Heisenberg exchange constants in Co(hcp) using MFT.
Test numerics and consistency of acoustic branch with a unit cell kernel."""

# General modules
import numpy as np

# Script modules
from ase.build import bulk

from gpaw import GPAW, PW, FermiDirac
from gpaw.response.mft import IsotropicExchangeCalculator
from gpaw.response.site_kernels import SphericalSiteKernels
from gpaw.response.heisenberg import calculate_FM_magnon_energies


def test_Co_hcp():
    # ---------- Inputs ---------- #

    # Part 1: Ground state calculation
    # Atomic configuration
    a = 2.5071
    c = 4.0695
    mm = 1.6
    # Ground state parameters
    xc = 'LDA'
    kpts = 6
    occw = 0.01
    nbands = 2 * (6 + 0)  # 4s + 3d + 0 empty shell bands
    ebands = 2 * 2  # extra bands for ground state calculation
    pw = 200
    conv = {'density': 1e-8,
            'forces': 1e-8,
            'bands': nbands}

    # Part 2: MFT calculation
    ecut = 150
    # Do high symmetry points of the hcp lattice
    q_qc = np.array([[0, 0, 0],              # Gamma
                     [0.5, 0., 0.],          # M
                     [1. / 3., 1 / 3., 0.],  # K
                     [0., 0., 0.5]           # A
                     ])

    # Use spherical site kernels in a radius range which should yield
    # stable results
    rc_pa = np.array([[1.0, 1.0], [1.1, 1.1], [1.2, 1.2]])

    # ---------- Script ---------- #

    # Part 1: Ground state calculation

    atoms = bulk('Co', 'hcp', a=a, c=c)
    atoms.set_initial_magnetic_moments([mm, mm])
    atoms.center()

    calc = GPAW(xc=xc,
                mode=PW(pw),
                kpts={'size': (kpts, kpts, kpts), 'gamma': True},
                occupations=FermiDirac(occw),
                convergence=conv,
                nbands=nbands + ebands,
                symmetry={'point_group': False},
                idiotproof=False,
                parallel={'domain': 1})

    atoms.calc = calc
    atoms.get_potential_energy()

    # Part 2: MFT calculation

    # Set up spherical site kernels
    positions = atoms.get_positions()
    sitekernels = SphericalSiteKernels(positions, rc_pa)
    # To do: Add unit cell site kernel XXX

    # Initialize the exchange calculator
    isoexch_calc = IsotropicExchangeCalculator(calc, ecut=ecut, nbands=nbands)

    # Allocate array for the exchange constants
    nq = len(q_qc)
    nsites = sitekernels.nsites
    npartitions = sitekernels.npartitions
    J_qabp = np.empty((nq, nsites, nsites, npartitions), dtype=complex)

    # Calcualate the exchange constants for each q-point
    for q, q_c in enumerate(q_qc):
        J_qabp[q] = isoexch_calc(q_c, sitekernels)

    # Calculate the magnon energy
    mm_ap = np.average(calc.get_magnetic_moments())\
        * np.ones((nsites, npartitions))
    mw_qnp = calculate_FM_magnon_energies(J_qabp, q_qc, mm_ap)
    mw_qnp = np.sort(mw_qnp, axis=1)  # Make sure the eigenvalues are sorted

    # Part 3: Compare results to test values
    print(mw_qnp[..., 0])
    print(mw_qnp[..., 1])
    print(mw_qnp[..., 2])
    assert 0

    # Part 4: Check self-consistency of results XXX
