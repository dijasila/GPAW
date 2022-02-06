"""Test for elph/selfenergy"""
import pytest

from ase.build import bulk
from ase.phonons import Phonons

from gpaw import GPAW
# from gpaw.mpi import world

from gpaw.elph import ElectronPhononMatrix, Selfenergy


@pytest.mark.elph
def test_selfenergy(module_tmp_path, elph_cache, supercell_cache):
    atoms = bulk('Li', crystalstructure='bcc', a=3.51, cubic=True)
    calc = GPAW(mode='lcao',
                basis='sz(dzp)',
                kpts={'size': (2, 2, 2), 'gamma': False},
                symmetry='off',
                txt='li_gs_nosym.txt')
    atoms.calc = calc
    atoms.get_potential_energy()

    elph_cache
    phonon = Phonons(atoms, name='elph', supercell=(2, 1, 1),
                     center_refcell=True)
    phonon.read()
    w_ql = phonon.band_structure(path_kc=calc.wfs.kd.bzk_kc, modes=False)
    print(calc.wfs.kd.bzk_kc)
    print(w_ql)
    # [[0.00124087  0.00044091  0.00132004  0.0421112   0.04212737  0.04218485]
    # [0.03031018  0.03031948  0.03041029  0.03041035  0.04326759  0.04327498]]

    supercell_cache
    elph = ElectronPhononMatrix(atoms, 'supercell', 'elph')
    elph.bloch_matrix(calc, savetofile=True, prefactor=False)

    se = Selfenergy(calc, w_ql)
    fan_sknn = se.fan_self_energy_sa()
    print(fan_sknn)
    assert fan_sknn.shape == (1, 8, 2, 2)
