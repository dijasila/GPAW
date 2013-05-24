import numpy as np
from gpaw.response.gw import GW

gw = GW(
        file='Si_groundstate.gpw',
        nbands=50,
        bands=np.array([2,3,4,5]), # must be the same as in exxfile
        kpoints=None,              # by default, these are all k-points in the irreducible Brillouin zone
        ecut=100.,
        ppa=True,
        txt='Si_gw-050bands.out'
       )

gw.get_QP_spectrum(exxfile='EXX_ecut100.pckl')
