from gpaw.elph.electronphonon import ElectronPhononCoupling
from ase.phonons import Phonons
from ase import Atoms
from gpaw import GPAW
import numpy as np

a = 0.90
atoms = Atoms('H',
              cell=np.diag([a, 2.1, 2.1]),
              positions=[[0, 0, 0]],
              pbc=(1, 0, 0))

atoms.center()
supercell = (2, 1, 1)
parameters = {'mode': 'lcao',
              'kpts': {'size': (2, 1, 1), 'gamma': True},
              'txt': None,
              'basis': 'dzp',
              'symmetry': {'point_group': False},
              'xc': 'PBE'}
elph_calc = GPAW(**parameters)
atoms.set_calculator(elph_calc)
atoms.get_potential_energy()
gamma_bands = elph_calc.wfs.kpt_u[0].C_nM

elph = ElectronPhononCoupling(atoms, elph_calc, supercell=supercell,
                              calculate_forces=True)
elph.run()

parameters['parallel'] = {'domain': 1}
elph_calc = GPAW(**parameters)
elph = ElectronPhononCoupling(atoms, calc=None, supercell=supercell)
elph.set_lcao_calculator(elph_calc)
elph.calculate_supercell_matrix(dump=1)

ph = Phonons(atoms=atoms, name='phonons', supercell=supercell, calc=None)
ph.read()
kpts = [[0, 0, 0]]
frequencies, modes = ph.band_structure(kpts, modes=True)

c_kn = np.array([[gamma_bands[0]]])
g_qklnn = elph.bloch_matrix(c_kn=c_kn, kpts=kpts, qpts=kpts, u_ql=modes)
