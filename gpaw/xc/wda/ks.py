def get_K_K(gd):
    from gpaw.utilities.tools import construct_reciprocal
    from ase.units import Bohr
    K2_K, _ = construct_reciprocal(gd)
    K2_K[0, 0, 0] = 0
    return K2_K**(1 / 2)
