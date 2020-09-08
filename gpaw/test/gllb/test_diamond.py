"""This calculation has the following structure.

1) Calculate the ground state of diamond.
2) Calculate the band structure of diamond in order to obtain
   accurate KS band gap.
3) Calculate the potential discontinuity using the ground state calculator
   and the accurate band gap.
4) Apply the discontinuity to CBM using the band structure calculator.

Compare to reference.
"""
import pytest
from ase.build import bulk
from ase.units import Ha
from gpaw import GPAW, Davidson, Mixer, restart
from gpaw.mpi import world


@pytest.mark.gllb
@pytest.mark.libxc
def test_gllb_diamond(in_tmp_dir):
    xc = 'GLLBSC'
    KS_gap_ref = 4.180237125868162
    QP_gap_ref = 5.469387490357182
    # M. Kuisma et. al, https://doi.org/10.1103/PhysRevB.82.115106
    #     C: KS gap 4.14 eV, QP gap 5.41eV, expt. 5.48 eV

    # Calculate ground state
    atoms = bulk('C', 'diamond', a=3.567)
    # We want sufficiently many grid points that the calculator
    # can use wfs.world for the finegd, to test that part of the code.
    calc = GPAW(h=0.2,
                kpts=(4, 4, 4),
                xc=xc,
                nbands=8,
                mixer=Mixer(0.5, 5, 10.0),
                parallel=dict(domain=min(world.size, 2), band=1),
                eigensolver=Davidson(niter=2),
                txt='gs.out')
    atoms.calc = calc
    atoms.get_potential_energy()

    # Calculate accurate KS-band gap from band structure
    bs_calc = calc.fixed_density(kpts={'path': 'GX', 'npoints': 12},
                                 symmetry='off',
                                 nbands=8,
                                 convergence={'bands': 6},
                                 eigensolver=Davidson(niter=4),
                                 txt='bs.out')
    # Get the accurate KS-band gap
    homolumo = bs_calc.get_homo_lumo()
    homo, lumo = homolumo
    KS_gap = lumo - homo
    print('KS gap', KS_gap)
    assert KS_gap == pytest.approx(KS_gap_ref, abs=1e-4)

    # Calculate the discontinuity potential with accurate band gap
    response = calc.hamiltonian.xc.xcs['RESPONSE']
    Dxc_pot = response.calculate_discontinuity_potential(homolumo=homolumo / Ha)

    # Redo the band structure calculation
    bs_response = bs_calc.hamiltonian.xc.xcs['RESPONSE']
    KS, dxc = bs_response.calculate_discontinuity(Dxc_pot)
    print('KS gap 2', KS)

    QP_gap = KS + dxc
    print('QP gap', QP_gap)
    assert QP_gap == pytest.approx(QP_gap_ref, abs=1e-4)
