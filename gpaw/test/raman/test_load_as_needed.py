import numpy as np
import pytest

from ase.phonons import Phonons
from gpaw import GPAW
from gpaw.elph.electronphonon import ElectronPhononCoupling
from gpaw.raman.elph import get_elph_matrix


class FakeEPC(ElectronPhononCoupling):
    """Fake ElectronPhononCoupling class to overwrite loading routine."""
    def __init__(self, fake_M_a, fake_nao_a, **params):
        ElectronPhononCoupling.__init__(self, **params)
        self.fake_M_a = fake_M_a
        self.fake_nao_a = fake_nao_a
        self.gx = np.random.random(size=[3, 1, 1, 1, 4, 4])

    def load_supercell_matrix_x(self, fname):
        index = int(fname.split('_')[-1].split('.')[0])
        g_sNNMM = self.gx[index]
        return g_sNNMM, self.fake_M_a, self.fake_nao_a


class FakePh(Phonons):
    """Fake Phonons object class to overwrite reading routine."""
    def read(self):
        natoms = len(self.indices)
        m_a = self.atoms.get_masses()
        self.m_inv_x = np.repeat(m_a[self.indices]**-0.5, 3)
        if self.D_N is None:
            self.D_N = np.random.random([3 * natoms, 3 * natoms])

@pyest.mark.disable
@pytest.mark.serial
def test_load_as_needed(gpw_files, tmp_path_factory):
    """Test of elph_matrix function as well as load_gx_as_needed feature."""
    calc = GPAW(gpw_files['bcc_li_lcao_wfs'])
    atoms = calc.atoms
    # Initialize calculator if necessary
    if not hasattr(calc.wfs, 'C_nM'):
        calc.wfs.set_positions
        calc.initialize_positions(atoms)

    # create phonon object
    phonon = FakePh(atoms)
    # create an electron-phonon object
    elph = FakeEPC([0], [4], atoms=atoms)
    elph.calc_lcao = calc  # set calc directly to circumvent some checks

    g1 = get_elph_matrix(atoms, calc, elph, phonon, dump=2,
                         load_gx_as_needed=True)
    g2 = get_elph_matrix(atoms, calc, elph, phonon, dump=2,
                         load_gx_as_needed=False)
    assert g2 == pytest.approx(g1)
