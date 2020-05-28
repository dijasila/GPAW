from ase.build import bulk
from gpaw import GPAW, PW


def test_metallic_gllbsc_fail(in_tmp_dir):
    repeat = 1
    atoms = bulk('Ag') * repeat
    k = 4 // repeat
    calc = GPAW(mode=PW(200),
                setups={'Ag': '11'},
                nbands=6 * repeat**3,
                xc='GLLBSC',
                kpts={'size': (k, k, k), 'gamma': False},
                txt='-')
    atoms.calc = calc
    try:
        atoms.get_potential_energy()
    except RuntimeError as e:
        assert 'GLLBSCM' in str(e), 'GLLBSCM not mentioned in the error'
    else:
        raise RuntimeError('GLLBSC should fail for metallic system')
