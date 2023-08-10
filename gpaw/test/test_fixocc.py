from ase.build import molecule
from ase.parallel import parprint
from gpaw import GPAW
from gpaw.cluster import Cluster
from gpaw.test import equal


def test_fixocc():
    h = 0.4
    box = 2
    nbands = 2
    txt = '-'
    txt = None

    H2 = Cluster(molecule('H2'))
    H2.minimal_box(box, h)
    convergence = {'energy': 0.01, 'eigenstates': 0.001, 'density': 0.01}

    base_kwargs = dict(
        mode='fd',
        h=h,
        nbands=nbands,
        convergence=convergence,
        txt=txt)

    if 1:
        # test ZeroKelvin vs FixedOccupations
        c = GPAW(**base_kwargs, occupations={'width': 0.0})
        H2.calc = c
        E_zk = H2.get_potential_energy()

        c = GPAW(**base_kwargs,
                 occupations=dict(name='fixed', numbers=[[1, 0]]))
        H2.calc = c
        E_fo = H2.get_potential_energy()
        parprint(E_zk, E_fo)
        equal(E_zk, E_fo, 1.e-10)

    if 1:
        # test spin-paired vs spin-polarized
        c = GPAW(**base_kwargs,
                 occupations={'name': 'fixed', 'numbers': [[0.5, 0.5]]})
        H2.calc = c
        E_ns = H2.get_potential_energy()
    if 1:
        c = GPAW(**base_kwargs,
                 spinpol=True,
                 occupations={'name': 'fixed', 'numbers': [[0.5, 0.5]] * 2})
        H2.calc = c
        E_sp = H2.get_potential_energy()
        parprint(E_ns, E_sp)
        equal(E_ns, E_sp, 1.e-6)
