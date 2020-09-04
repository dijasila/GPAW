"""This calculation has the following structure.

1) Calculate the ground state of Diamond.
2) Calculate the band structure of diamond in order to obtain accurate KS
   band gap for Diamond.
3) Calculate ground state again, and calculate the potential discontinuity
   using accurate band gap.
4) Calculate band structure again, and apply the discontinuity to CBM.

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
                parallel=dict(domain=min(world.size, 2),
                              band=1),
                eigensolver=Davidson(niter=2),
                txt='1.out')
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('Cgs.gpw')

    # Calculate accurate KS-band gap from band structure
    calc = calc.fixed_density(kpts={'path': 'GX', 'npoints': 12},
                              symmetry='off',
                              nbands=8,
                              convergence={'bands': 6},
                              eigensolver=Davidson(niter=4),
                              txt='2.out')
    # Get the accurate KS-band gap
    homolumo = calc.get_homo_lumo()
    homo, lumo = homolumo
    KS_gap = lumo - homo
    print('KS gap', KS_gap)
    assert KS_gap == pytest.approx(KS_gap_ref, abs=1e-4)

    # Redo the ground state calculation
    calc = GPAW(h=0.2, kpts=(4, 4, 4), xc=xc, nbands=8,
                mixer=Mixer(0.5, 5, 10.0),
                eigensolver=Davidson(niter=4),
                txt='3.out')
    atoms.calc = calc
    atoms.get_potential_energy()
    # And calculate the discontinuity potential with accurate band gap
    response = calc.hamiltonian.xc.xcs['RESPONSE']
    response.calculate_delta_xc(homolumo=homolumo / Ha)
    calc.write('CGLLBSC.gpw')

    # Redo the band structure calculation
    calc = calc.fixed_density(kpts={'path': 'GX', 'npoints': 12},
                              symmetry='off',
                              convergence={'bands': 6},
                              eigensolver=Davidson(niter=4),
                              txt='4.out')
    response = calc.hamiltonian.xc.xcs['RESPONSE']
    KS, dxc = response.calculate_delta_xc_perturbation()
    print('KS gap 2', KS)

    QP_gap = KS + dxc
    print('QP gap', QP_gap)
    assert QP_gap == pytest.approx(QP_gap_ref, abs=1e-4)
