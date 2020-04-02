from typing import NamedTuple, Dict, List, Any

import numpy as np

from gpaw.utilities import unpack2, unpack, packed_index


class PAWThings(NamedTuple):
    VC_aii: Dict[int, np.ndarray]
    VV_aii: Dict[int, np.ndarray]  # distributed over comm
    Delta_aiiL: List[np.ndarray]
    comm: Any


def calculate_paw_stuff(dens, setups):
    nspins = dens.nspins
    VV_saii = [{} for s in range(nspins)]
    for a, D_sp in dens.D_asp.items():
        data = setups[a]
        for VV_aii, D_p in zip(VV_saii, D_sp):
            D_ii = unpack2(D_p) * (nspins / 2)
            VV_ii = pawexxvv(data, D_ii)
            VV_aii[a] = VV_ii

    Delta_aiiL = []
    VC_aii = {}
    for a, data in enumerate(setups):
        Delta_aiiL.append(data.Delta_iiL)
        VC_aii[a] = unpack(data.X_p)

    return [PAWThings(VC_aii, VV_aii, Delta_aiiL, dens.gd.comm)
            for VV_aii in VV_saii]


def pawexxvv(atomdata, D_ii):
    """PAW correction for valence-valence EXX energy."""
    ni = len(D_ii)
    V_ii = np.empty((ni, ni))
    for i1 in range(ni):
        for i2 in range(ni):
            V = 0.0
            for i3 in range(ni):
                p13 = packed_index(i1, i3, ni)
                for i4 in range(ni):
                    p24 = packed_index(i2, i4, ni)
                    V += atomdata.M_pp[p13, p24] * D_ii[i3, i4]
            V_ii[i1, i2] = V
    return V_ii
