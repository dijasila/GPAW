"""
Test the born charges script on bulk silicon

Requirements are that the system is gapped.
Relaxed .gpw file does not have symmetry.

The borncharges script implements symmetry = off, displaces the atoms
in some direction and then computes the polarization from the Berry phases
Note that for the test, we only displace one atom relative to the other 
in the positive x direction.

"""

import pytest
import numpy as np
from gpaw import GPAW

from os.path import splitext, isfile
from os import remove
from glob import glob

from gpaw.mpi import world
from gpaw.berryphase import get_polarization_phase

from ase.units import Bohr
from ase.build import bulk

def test_born(in_tmp_dir, gpw_files):

    calc = GPAW(gpw_files['si_pw'])
    borncharges_test(calc)

def get_wavefunctions_test(atoms, name, params):
    params['symmetry'] = {'point_group': False,
                          'time_reversal': False}
    tmp = splitext(name)[0]
    atoms.calc = GPAW(txt=tmp + '.txt', **params)
    atoms.get_potential_energy()
    atoms.calc.write(name, 'all')
    return atoms.calc

def borncharges_test(calc, delta=0.01):

    params = calc.parameters
    atoms = calc.atoms
    cell_cv = atoms.get_cell() / Bohr
    vol = abs(np.linalg.det(cell_cv))
    sym_a = atoms.get_chemical_symbols()

    # List for atomic indices
    indices = [0]          #test only computes one atom

    pos_av = atoms.get_positions()
    avg_v = np.sum(pos_av, axis=0) / len(pos_av)
    pos_av -= avg_v
    atoms.set_positions(pos_av)
    Z_avv = []
    norm_c = np.linalg.norm(cell_cv, axis=1)
    proj_cv = cell_cv / norm_c[:, np.newaxis]

    B_cv = np.linalg.inv(cell_cv).T * 2 * np.pi
    area_c = np.zeros((3,), float)
    area_c[[2, 1, 0]] = [np.linalg.norm(np.cross(B_cv[i], B_cv[j]))
                         for i in range(3)
                         for j in range(3) if i < j]

    for a in indices:
        phase_scv = np.zeros((2, 3, 3), float)
        for v in range(1):                            #test only computes displacements in one dir
            for s, sign in enumerate([-1, 1]):
                if world.rank == 0:
                    print(sym_a[a], a, v, s)
                atoms.positions = pos_av
                atoms.positions[a, v] = pos_av[a, v] + sign * delta
                prefix = f'born-{delta}-{a}{"xyz"[v]}{" +-"[sign]}'
                name = f'{prefix}.gpw'

                calc = get_wavefunctions_test(atoms, name, params)
                try:
                    phase_c = get_polarization_phase(name)
                except ValueError:
                    calc = get_wavefunctions_test(atoms, name, params)
                    phase_c = get_polarization_phase(name)

                phase_scv[s, :, v] = phase_c

        dphase_cv = (phase_scv[1] - phase_scv[0])
        dphase_cv -= np.round(dphase_cv / (2 * np.pi)) * 2 * np.pi
        dP_cv = (area_c[:, np.newaxis] / (2 * np.pi)**3 * dphase_cv)
        dP_vv = np.dot(proj_cv.T, dP_cv)
        Z_vv = dP_vv * vol / (2 * delta / Bohr)
        Z_avv.append(Z_vv)

    ref_value = 0.00010643781178941813

    print('Ref value of Z_0xx', ref_value)
    print('Z_0xx this calculation', Z_avv[0][0][0])

    err1 = abs(Z_avv[0][0][0] - ref_value)
    print('Error', err1)
    assert err1 < 1e-5, err1
