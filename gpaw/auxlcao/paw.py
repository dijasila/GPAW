from gpaw.utilities import unpack2, packed_index, pack2, pack
import numpy as np

def paw_exx_correction(setup, D_sp, dH_sp, exx_fraction):
    D_ii = unpack2(D_sp[0]) / 2 # Check 1 or 2
    ni = len(D_ii)
    V_ii = np.empty((ni, ni))
    for i1 in range(ni):
        for i2 in range(ni):
            V = 0.0
            for i3 in range(ni):
                p13 = packed_index(i1, i3, ni)
                for i4 in range(ni):
                    p24 = packed_index(i2, i4, ni)
                    V += setup.ri_M_pp[p13, p24] * D_ii[i3, i4]
            V_ii[i1, i2] = 2*V
    V_p = pack2(V_ii)
    dH_sp[0][:] += (-V_p - setup.ri_X_p) * exx_fraction
    evv = -exx_fraction * np.dot(V_p, D_sp[0]) / 2
    evc = -exx_fraction * np.dot(D_sp[0], setup.ri_X_p)
    return evv + evc, 0.0

