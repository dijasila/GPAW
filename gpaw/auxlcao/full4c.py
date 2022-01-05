import numpy as np
from gpaw.auxlcao.algorithm import RIAlgorithm

class Full4C(RIAlgorithm):
    def __init__(self, exx_fraction=None, screening_omega=None):
        RIAlgorithm.__init__(self, 'Full4C debug', exx_fraction, screening_omega)

    def set_positions(self, spos_ac):
        RIAlgorithm.set_positions(self, spos_ac)
        self.spos_ac = spos_ac

    def nlxc(self, 
             H_MM:np.ndarray,
             dH_asp:Dict[int,np.ndarray],
             wfs,
             kpt) -> Tuple[float, float, float]:
        evc = 0.0
        evv = 0.0
        ekin = 0.0
        ekin = -2*evv -evc
        return evv, evc, ekin 

    def get_description(self):
        return 'Debug evaluation of full 4 center matrix elements'
