import numpy as np
import pytest

from ase.phonons import Phonons
from ase.utils.filecache import MultiFileJSONCache
from gpaw import GPAW
from gpaw.raman.elph import EPC


class FakeEPC(EPC):
    """Fake ElectronPhononCoupling class to overwrite loading routine."""
    def __init__(self, fake_basis, natom, **params):
        EPC.__init__(self, **params)
        # self.gx = np.random.random(size=[3, 1, 1, 1, 4, 4])
        gx = np.arange(3 * natom * 4 * 4).reshape([3 * natom, 1, 1, 1, 4, 4])

        # Fake supercell cache
        self.supercell_cache = MultiFileJSONCache('supercell')
        with self.supercell_cache.lock('basis') as handle:
            if handle is not None:
                handle.save(fake_basis)
        for x in range(3 * natom):
            with self.supercell_cache.lock(str(x)) as handle:
                if handle is None:
                    continue
                handle.save(gx[x])


class FakePh(Phonons):
    """Fake Phonons object class to overwrite reading routine."""
    def read(self):
        natoms = len(self.indices)
        m_a = self.atoms.get_masses()
        self.m_inv_x = np.repeat(m_a[self.indices]**-0.5, 3)
        if self.D_N is None:
            # self.D_N = np.random.random([3 * natoms, 3 * natoms])
            self.D_N = np.ones([3 * natoms, 3 * natoms])


@pytest.mark.serial
def test_elph_matrix(gpw_files, tmp_path_factory):
    """Test of elph_matrix function as well as load_gx_as_needed feature."""
    calc = GPAW(gpw_files['bcc_li_lcao_wfs'])
    atoms = calc.atoms
    # Initialize calculator if necessary
    if not hasattr(calc.wfs, 'C_nM'):
        calc.wfs.set_positions
        calc.initialize_positions(atoms)

    for kpt in calc.wfs.kpt_u:
        print(kpt.eps_n)

    # create phonon object
    phonon = FakePh(atoms)
    # create an electron-phonon object
    elph = FakeEPC(fake_basis={'M_a': [0], 'nao_a': [4]}, natom=1,
                   atoms=atoms)
    elph.calc_lcao = calc  # set calc directly to circumvent some checks

    g_sqklnn = elph.get_elph_matrix(calc, phonon, savetofile=False)
    assert g_sqklnn.shape == (1, 1, 4, 3, 4, 4)

    # quick check of phonon object
    phonon.read()
    frequencies, modes = phonon.band_structure([[0., 0., 0.]], modes=True)
    modes = modes.reshape(3, 3).real
    assert modes[2] == pytest.approx([0.21915917, 0.21915917, 0.21915917])

    # this should always be the same, as long as CnM doesn't change
    # there are plenty of generate bands. will lead of different results on
    # different machines for some bands
    tmp = g_sqklnn[0, 0, :, 2].real
    print(tmp)
    # bands 1-3 degenerate for this kpoint, don't use
    assert tmp[0, 0, 0] == pytest.approx(8.57864400e+01)
    # bands 2-3 degenerate for this kpoint, don't use
    assert tmp[1, 0, 0] == pytest.approx(2.49398305e+02)
    # absolute value because of random phase
    assert abs(tmp[1, 0, 1]) == pytest.approx(4.15645630e+02)
    assert tmp[1, 1, 1] == pytest.approx(1.41967585e+04)
