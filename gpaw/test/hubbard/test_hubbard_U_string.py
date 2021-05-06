from gpaw.setup import parse_hubbard_string
from ase.units import Hartree


def test_Hubbard_U_string():
    # Format of string = type:l,U,scale;l,U,scale
    # U has an internal unit conversion to atomic units

    # Simple single correction
    str1 = ':d,5,0'
    expected1 = ('paw', [2], [5.0 / Hartree], [False])
    assert expected1 == parse_hubbard_string(str1)

    # Simple single correction without scale
    str1 = ':d,5'
    expected1 = ('paw', [2], [5.0 / Hartree], [True])
    assert expected1 == parse_hubbard_string(str1)

    # Multi correction
    str2 = 'ae:s,3,0;p,2.5,1;f,0.5,0'
    expected2 = ('ae',
                 [0, 1, 3],
                 [3.0 / Hartree, 2.5 / Hartree, 0.5 / Hartree],
                 [False, True, False])
    assert expected2 == parse_hubbard_string(str2)
