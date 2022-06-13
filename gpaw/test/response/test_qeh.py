import pytest
import numpy as np
from gpaw.response.g0w0 import G0W0
from pathlib import Path
from gpaw.mpi import world
from gpaw.response.df import DielectricFunction
from gpaw.response.qeh import BuildingBlock, check_building_blocks
from gpaw.response.qeh import interpolate_building_blocks

"""
xxx QEH module seem to require at least 6x6x1 kpoints.
    -this should be investigated
xxx Often fails with unreadable errors in interpolation.
    -arrays should be checked with assertions and readable errors
xxx add_intraband fails with NotImplementedError in dielctric 
    function. -Implement or remove option????
xxx isotropic_q = False is temporarily turned off. However, 
    most features require isotropic_q = True anyway.
    Should we remove the option or should we expand QEH to handle
    non-isotropic q?
"""

def dielectric(calc,domega,omega2):
    diel = DielectricFunction(calc=calc,
                              frequencies = {'type': 'nonlinear',
                                             'omegamax': 10,
                                             'domega0': domega,
                                             'omega2': omega2},
                              nblocks=1,
                              ecut=10,
                              truncation='2D')
    return diel


@pytest.mark.response
def test_basics(in_tmp_dir, gpw_files):
    qeh = pytest.importorskip('qeh')
    Heterostructure = qeh.Heterostructure    

    df = dielectric(gpw_files['graphene_pw_wfs'], 0.2, 0.6)
    df2 = dielectric(gpw_files['mos2_pw_wfs'], 0.1, 0.5)

    #Testing to compute building block
    bb1 = BuildingBlock('graphene', df)
    bb2 = BuildingBlock('mos2', df2)    
    bb1.calculate_building_block()
    bb2.calculate_building_block()
    
    #Test building blocks are on different grids
    are_equal=check_building_blocks(['mos2','graphene'])
    assert not are_equal

    #testing to interpolate
    interpolate_building_blocks(BBfiles=['graphene'], BBmotherfile='mos2')
    are_equal=check_building_blocks(['mos2_int','graphene_int'])
    assert are_equal

    #test qeh interface
    HS = Heterostructure(structure=['mos2_int', 'graphene_int'],d=[5],wmax=0,d0=5)
    chi = HS.get_chi_matrix()
    assert np.amax(chi) == pytest.approx(0.019456648867161096-0.00023954749821020183j)
        
    #test to interpolate to grid and actual numbers
    q_grid = np.array([0, 0.1])
    w_grid = np.array([0, 0.1])
    bb2.interpolate_to_grid(q_grid=q_grid, w_grid=w_grid)
    data = np.load('mos2_int-chi.npz')
    assert np.allclose(data['omega_w'],np.array([0.,0.00367493]))

    monopole = np.array([[-7.21522101e-10+3.66116609e-23j, -7.22838580e-10-5.53789786e-12j],
                         [-7.32406423e-03-8.85107345e-21j, -7.32900925e-03-2.03292271e-05j]])
    assert np.allclose(data['chiM_qw'],monopole)

    dipole=np.array([[-0.19421447+7.45622655e-19j, -0.19436547-6.15346284e-04j],
                     [-0.20438539+7.71005749e-19j, -0.20455297-6.83427250e-04j]])
    assert np.allclose(data['chiD_qw'],dipole)
