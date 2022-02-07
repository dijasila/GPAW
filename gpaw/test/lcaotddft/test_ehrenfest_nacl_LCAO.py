from ase import Atoms
from gpaw import GPAW
from gpaw.tddft.ehrenfest import EhrenfestVelocityVerlet
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.test import equal


def test_tddft_ehrenfest_nacl_LCAO(in_tmp_dir):
    """ Simple test for LCAO Ehrenfest dynamics """
    # extend this test for parallel calculations and other basis sets
    d = 4.0
    atoms = Atoms('NaCl', [(0, 0, 0), (0, 0, d)])
    atoms.center(vacuum=4.5)

    gs_calc = GPAW(mode='lcao', basis='sz(dzp)', xc='LDA',
                   setups={'Na': '1'}, txt='out.txt')
    atoms.calc = gs_calc
    atoms.get_potential_energy()

    gs_calc.write('nacl_gs.gpw', 'all')

    td_calc = LCAOTDDFT('nacl_gs.gpw', propagator='edsicn', txt='out_td.txt',
                        PLCAO_flag=False, Ehrenfest_force_flag=False, S_flag=True)
    td_calc.tddft_init()
    evv = EhrenfestVelocityVerlet(td_calc)

    i = 0
    evv.get_energy()
    r = evv.positions[1][2] - evv.positions[0][2]
    print('E = ', i, r, evv.Etot, evv.Ekin, evv.e_coulomb)

    for i in range(5):
        evv.propagate(1.0)
        evv.get_energy()
        r = evv.positions[1][2] - evv.positions[0][2]
        print('E = ', i + 1, r, evv.Etot, evv.Ekin, evv.e_coulomb)

    equal(r, 7.558904486952679, 1e-8)
    equal(evv.Etot, -0.04631107718681388, 1e-8)
    # equal(evv.Etot, -0.08523047898852674, 1e-8)
