from math import sqrt

import ase.io
import numpy as np
from ase import Atoms
from ase.build import add_adsorbate, fcc100, fcc111, molecule
from ase.io import read
from ase.lattice.cubic import FaceCenteredCubic
from gpaw import PW, FermiDirac
from gpaw.utilities import h2gpts


def _create_test_systems():
    systems = {}
    for name, obj in globals().items():
        if callable(obj) and not name.startswith('_'):
            atoms, params = obj()
            systems[name] = (atoms, params)
    return systems


def a5():
    atoms = ase.io.read('alpra_Mg.CONTCAR')
    atoms.pbc = True
    return atoms, dict(xc='vdW-DF',
                       occupations={'name': 'fermi-dirac', 'width': 0.1})


def ni100():
    atoms = fcc100(symbol='Ni', size=(1, 1, 9), a=3.52, vacuum=5.5)
    atoms.set_initial_magnetic_moments([0.6] * len(atoms))
    gpts = h2gpts(0.18, atoms.cell, idiv=8)
    return atoms, dict(gpts=gpts,
                       kpts=(8, 8, 1),
                       xc='PBE')


def biimtf():
    atoms = read('biimtf.xyz')
    atoms.center(vacuum=5)
    return atoms, dict(h=0.16,
                       charge=+1,
                       occupations=FermiDirac(0.05),
                       xc='RPBE')


def opt111b():
    atoms = fcc111('Pt', (2, 2, 6), a=4.00, vacuum=10.0)
    add_adsorbate(atoms, 'O', 2.0, 'fcc')
    return atoms, dict(mode='pw',
                       kpts=(8, 8, 1),
                       xc='RPBE')


def scsz():
    # the system losing magnetic moment
    atoms = read('ScSZ.xyz')
    atoms.set_cell([[7.307241, 0., 0.],
                   [0., 12.656514, 0.],
                   [0., 0., 19.]],
                   scale_atoms=False)
    atoms.center(axis=2)
    magmoms = [0.0 for n in range(len(atoms))]
    for n, a in enumerate(atoms):
        if a.symbol == 'Ni':
            magmoms[n] = 0.6
    atoms.set_initial_magnetic_moments(magmoms)

    atoms.pbc = (True, True, False)

    return atoms, dict(h=0.2,
                       kpts=(2, 1, 1),
                       xc='PBE',
                       maxiter=100,
                       txt='ScSZ.txt')


def tio2v2o5():
    atoms = Atoms('(O2Ti4O6)3V2O5',
                  [(8.3411, 1.9309, 9.2647),
                   (3.2338, 0.0854, 9.4461),
                   (1.5783, 0.1417, 8.4327),
                   (6.6126, 2.0588, 8.2126),
                   (9.4072, 2.0891, 7.6857),
                   (4.2748, 0.1256, 7.9470),
                   (9.9477, 0.2283, 7.3281),
                   (4.7391, 2.0618, 7.6111),
                   (7.7533, 2.1354, 6.7529),
                   (2.6347, 0.2585, 6.9403),
                   (6.2080, 0.1462, 8.6280),
                   (0.9517, 1.9798, 8.9832),
                   (8.5835, 6.2280, 9.4350),
                   (3.1930, 3.9040, 9.7886),
                   (1.4899, 4.0967, 8.8898),
                   (6.6167, 5.8865, 8.8463),
                   (9.3207, 5.9258, 7.7482),
                   (4.1984, 3.9386, 8.2442),
                   (9.9075, 3.9778, 7.6337),
                   (4.7626, 5.8322, 8.0051),
                   (7.6143, 5.6963, 7.1276),
                   (2.5760, 3.9661, 7.3115),
                   (6.2303, 3.8223, 8.8067),
                   (1.1298, 5.9913, 8.6968),
                   (8.3845, 9.7338, 9.1214),
                   (3.1730, 7.9593, 9.3632),
                   (1.5914, 7.8120, 8.2310),
                   (6.7003, 9.7064, 8.1528),
                   (9.3943, 9.7202, 7.6037),
                   (4.3168, 7.7857, 7.9666),
                   (9.9045, 7.7968, 7.2716),
                   (4.7772, 9.7015, 7.4648),
                   (7.7314, 9.7221, 6.6253),
                   (2.7673, 7.6929, 6.8222),
                   (6.2358, 7.8628, 8.6557),
                   (1.0528, 9.7017, 8.5919),
                   (8.4820, 5.0952, 11.4981),
                   (9.7787, 2.0447, 11.0800),
                   (6.4427, 5.6315, 10.7415),
                   (10.7389, 0.4065, 11.8697),
                   (8.3109, 3.0048, 12.4083),
                   (10.4702, 4.1612, 10.6543),
                   (8.9827, 6.3884, 13.0109)],
                  cell=[10.152054, 11.430000, 18.295483],
                  pbc=[1, 1, 0])

    return atoms, dict(h=0.20,
                       kpts=(2, 2, 1),
                       xc='RPBE')


def pt_h2o():
    atoms = read('Pt_H2O.xyz')
    atoms.set_cell([[8.527708, 0, 0],
                   [0, 4.923474, 0],
                   [0, 0, 16]],
                   scale_atoms=False)
    atoms.center(axis=2)

    atoms.pbc = (True, True, False)

    return atoms, dict(h=0.20,
                       kpts=(2, 4, 1),
                       xc='RPBE',
                       poissonsolver={'dipolelayer': 'xy'},
                       basis='dzp',
                       maxiter=200,
                       txt='Pt_H2O.txt')


def lih():
    atoms = molecule('LiH')
    atoms.cell = [12, 12.01, 12.02]
    atoms.center()
    return atoms, dict(mode=PW(400),
                       xc='PBE')


def gs_small():
    NBN = 7
    NGr = 7
    a = 2.5
    c = 3.22

    GR = Atoms(symbols='C2',
               positions=[(0.5 * a, -sqrt(3) / 6 * a, 0.0),
                          (0.5 * a, +sqrt(3) / 6 * a, 0.0)],
               cell=[(0.5 * a, -0.5 * 3**0.5 * a, 0),
                     (0.5 * a, +0.5 * 3**0.5 * a, 0), (0.0, 0.0, c * 2.0)])
    GR.set_pbc((True, True, True))

    GR2 = GR.copy()
    cell = GR2.get_cell()
    uv = cell[0] - cell[1]
    uv = uv / np.sqrt(np.sum(uv**2.0))
    dist = np.array([0.5 * a, -sqrt(3) / 6 * a]) - np.array(
        [0.5 * a, +sqrt(3) / 6 * a])
    dist = np.sqrt(np.sum(dist**2.0))
    GR2.translate(uv * dist)

    BN = Atoms(symbols='BN',
               positions=[(0.5 * a, -sqrt(3) / 6 * a, 0.0),
                          (0.5 * a, +sqrt(3) / 6 * a, 0.0)],
               cell=[(0.5 * a, -0.5 * 3**0.5 * a, 0),
                     (0.5 * a, +0.5 * 3**0.5 * a, 0), (0.0, 0.0, c * 2.0)])
    BN.set_pbc((True, True, True))

    NB = Atoms(symbols='NB',
               positions=[(0.5 * a, -sqrt(3) / 6 * a, 0.0),
                          (0.5 * a, +sqrt(3) / 6 * a, 0.0)],
               cell=[(0.5 * a, -0.5 * 3**0.5 * a, 0),
                     (0.5 * a, +0.5 * 3**0.5 * a, 0), (0.0, 0.0, c * 2.0)])
    NB.set_pbc((True, True, True))

    GR2.translate([0, 0, c])
    NB.translate([0, 0, (NGr + 1.0 * (1 - NGr % 2)) * c] +
                 uv * dist * (NGr % 2))
    BN.translate([0, 0, (NGr + 1.0 * (NGr % 2)) * c] + uv * dist * (NGr % 2))

    GRBN = (GR * (1, 1, (NGr % 2 + NGr // 2)) + GR2 * (1, 1, (NGr // 2)) + NB *
            (1, 1, (NBN // 2 + (NBN % 2) * (NGr % 2))) + BN *
            (1, 1, (NBN // 2 + (NBN % 2) * (1 - NGr % 2))))
    BNNB = BN + NB
    Graphite = GR + GR2

    Graphite.set_pbc((True, True, True))
    old_cell = GR.get_cell()
    old_cell[2, 2] = 2 * c
    Graphite.set_cell(old_cell)

    BNNB.set_pbc((True, True, True))
    old_cell = BN.get_cell()
    old_cell[2, 2] = 2 * c
    BNNB.set_cell(old_cell)
    BNNB.center()

    GRBN.set_pbc((True, True, True))
    old_cell = BN.get_cell()
    old_cell[2, 2] = (NGr + NBN) * c
    GRBN.set_cell(old_cell)

    atoms = GRBN

    return atoms, dict(h=0.18,
                       mode=PW(600),
                       kpts=(29, 29, 1),
                       xc='PBE',
                       occupations=FermiDirac(0.01),
                       parallel={'band': 1})


def na2():
    atoms = molecule('Na2')
    atoms.cell = [12, 12.01, 12.02]
    atoms.center()
    return atoms, dict(mode=PW(400),
                       xc='PBE')


def opt111():
    atoms = fcc111('Pt', (2, 2, 6), a=4.00, vacuum=10.0)
    add_adsorbate(atoms, 'O', 2.5, 'fcc')
    return atoms, dict(mode='pw',
                       kpts=(8, 8, 1),
                       xc='RPBE')


def pt13():
    element = 'Pt'
    atoms = FaceCenteredCubic(symbol=element,
                              size=(2, 2, 2),
                              directions=[[1, 1, 0],
                                          [-1, 1, 0],
                                          [0, 0, 1]])
    del atoms[4]
    del atoms[3]
    del atoms[2]

    kpts = (8, 8, 4)
    ecut = 800
    xc1 = 'PBE'
    return atoms, dict(mode=PW(ecut=ecut),
                       kpts=kpts,
                       xc=xc1)


def cuni():
    atoms = Atoms('CuNi', magmoms=[0, 0.055])
    atoms.positions[1, 2] = 9.2920211200 - 7.4999788800
    atoms.center(vacuum=7.5)
    return atoms, dict(mode=dict(name='pw', ecut=40 * 13.6),
                       xc='PBE',
                       occupations=FermiDirac(width=0.003))


def gqd_triangle_o():
    atoms = Atoms(symbols='C22H12O',
                  cell=[27.32059936, 20.76985686, 27.31710493],
                  positions=[[10.73088624, 8.89837485, 12.55791070],
                             [10.05301739, 8.89679181, 13.79134457],
                             [10.75109359, 8.93020267, 14.99413278],
                             [12.87455486, 8.96133088, 11.26154178],
                             [12.15171434, 8.95142166, 12.50351937],
                             [12.88537206, 8.99960120, 13.73593471],
                             [12.18470434, 8.97912415, 15.00615124],
                             [12.93041976, 9.00897627, 16.21171088],
                             [15.00139248, 9.04826138, 9.94920307],
                             [14.26001879, 9.03723869, 11.21347057],
                             [15.00007481, 9.12042787, 12.44746145],
                             [14.31001394, 9.07881510, 13.70577598],
                             [15.05204214, 9.11724418, 14.94015062],
                             [14.35398396, 9.06815937, 16.20191803],
                             [15.12643840, 9.08256916, 17.40641817],
                             [16.34154900, 9.19271021, 9.89960611],
                             [17.21352972, 9.46240940, 11.11599421],
                             [16.43580351, 9.23675012, 12.44682100],
                             [17.15060465, 9.27560508, 13.63423194],
                             [16.48917583, 9.19801051, 14.90782943],
                             [17.19566489, 9.21120025, 16.13019394],
                             [16.51583146, 9.14444384, 17.35895034],
                             [8.95901044, 8.86943060, 13.79990435],
                             [10.22073262, 8.92555557, 15.94924199],
                             [12.40194675, 8.99027108, 17.16999696],
                             [14.60100969, 9.04207920, 18.36575973],
                             [17.08913360, 9.14288668, 18.29186733],
                             [18.28620375, 9.27200790, 16.10970058],
                             [18.23171084, 9.40588926, 13.60237819],
                             [18.05103187, 8.72914292, 11.10056947],
                             [16.84288323, 9.24967730, 8.93522359],
                             [14.39939659, 8.96835388, 9.02869781],
                             [12.30013204, 8.91463681, 10.32539251],
                             [10.17280943, 8.86784643, 11.61496353],
                             [17.66988350, 10.75156430, 11.03936397]])
    atoms.set_initial_magnetic_moments([0] * 34 + [1])
    return atoms, {}


def cofe2o4():
    atoms = Atoms(symbols='CoFe2O4',
                  magmoms=[3, 3, 3, 0, 0, 0, 0],
                  cell=[19.25300000, 20.91400000, 21.31300000],
                  positions=[[9.26470177, 10.75627983, 9.27350608],
                             [10.18629541, 11.21886515, 11.29618968],
                             [9.56307536, 9.00478168, 10.73991807],
                             [8.99650931, 9.87814613, 12.50179659],
                             [9.24889533, 8.89915474, 8.99153197],
                             [10.40644251, 9.61524884, 12.32000271],
                             [9.80651995, 12.33897925, 9.84687046]])
    return atoms, {}


def isa():
    return read('MoC2-graphene-4N-1Co-clean.xyz'), {}
